#!/usr/bin/env python3
"""Run one DeepSeek-V4-Flash prefill directly through vLLM-Ascend.

This deliberately excludes verl, Ray, actor training, teacher inference, and
distillation.  Run one prompt length per process so a hanging case does not
hide the last successful boundary test.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from uuid import uuid4

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.v1.engine.async_llm import AsyncLLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    prompt_source = parser.add_mutually_exclusive_group(required=True)
    prompt_source.add_argument("--prompt-length", type=int)
    prompt_source.add_argument(
        "--prompt-token-ids-file",
        help="JSON file containing the exact list of prompt token IDs to replay.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8196)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument(
        "--disable-dsa-cp",
        action="store_true",
        help="Disable the DSV4 DSA context-parallel implementation for comparison.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable graph execution for comparison.",
    )
    parser.add_argument(
        "--async-engine",
        action="store_true",
        help="Use AsyncLLM/AsyncMPClient, matching the verl rollout server.",
    )
    return parser.parse_args()


def make_prompt_token_ids(model_path: str, prompt_length: int) -> list[int]:
    if prompt_length < 1:
        raise ValueError("--prompt-length must be positive")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    body_ids = tokenizer.encode("Hello", add_special_tokens=False)
    if not body_ids:
        raise RuntimeError("Tokenizer returned no token for the diagnostic text")

    prefix = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    body_length = prompt_length - len(prefix)
    if body_length < 0:
        prefix = prefix[:prompt_length]
        body_length = 0
    return prefix + [body_ids[-1]] * body_length


def load_prompt_token_ids(args: argparse.Namespace) -> list[int]:
    if args.prompt_token_ids_file:
        with open(args.prompt_token_ids_file, encoding="utf-8") as f:
            prompt_token_ids = json.load(f)
        if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
            raise ValueError("Prompt token file must contain a non-empty JSON list")
        if not all(isinstance(token_id, int) for token_id in prompt_token_ids):
            raise ValueError("Every entry in the prompt token file must be an integer")
        return prompt_token_ids
    return make_prompt_token_ids(args.model, args.prompt_length)


def main() -> None:
    args = parse_args()
    prompt_token_ids = load_prompt_token_ids(args)

    print(
        "[diagnose] engine init: "
        f"model={args.model}, prompt_length={len(prompt_token_ids)}, "
        f"tp={args.tensor_parallel_size}, chunk={args.max_num_batched_tokens}, "
        f"dsa_cp={not args.disable_dsa_cp}, eager={args.enforce_eager}, "
        f"async_engine={args.async_engine}, temperature={args.temperature}, "
        f"top_p={args.top_p}, top_k={args.top_k}",
        flush=True,
    )
    init_start = time.monotonic()
    engine_kwargs = {
        "model": args.model,
        "trust_remote_code": True,
        "tokenizer_mode": "deepseek_v4",
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": "bfloat16",
        "load_format": "safetensors",
        "max_model_len": args.max_model_len,
        "max_num_seqs": 1,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_chunked_prefill": True,
        "enable_expert_parallel": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "block_size": 128,
        "enforce_eager": args.enforce_eager,
        "compilation_config": {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4],
        },
        "additional_config": {
            "enable_dsa_cp": not args.disable_dsa_cp,
            "weight_nz_mode": 0,
        },
    }

    if args.async_engine:
        asyncio.run(run_async(engine_kwargs, prompt_token_ids, init_start, args))
        return

    llm = LLM(**engine_kwargs)
    print(f"[diagnose] engine ready in {time.monotonic() - init_start:.1f}s", flush=True)

    print("[diagnose] generate start", flush=True)
    generate_start = time.monotonic()
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_token_ids)],
        SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=1,
        ),
        use_tqdm=False,
    )
    generated_ids = outputs[0].outputs[0].token_ids
    print(
        f"[diagnose] generate done in {time.monotonic() - generate_start:.1f}s: " f"token_ids={generated_ids}",
        flush=True,
    )


async def run_async(
    engine_kwargs: dict[str, object],
    prompt_token_ids: list[int],
    init_start: float,
    args: argparse.Namespace,
) -> None:
    engine_args = AsyncEngineArgs(**engine_kwargs)
    engine = AsyncLLM.from_engine_args(engine_args)
    print(f"[diagnose] engine ready in {time.monotonic() - init_start:.1f}s", flush=True)

    print("[diagnose] generate start", flush=True)
    generate_start = time.monotonic()
    final_output = None
    try:
        async for output in engine.generate(
            TokensPrompt(prompt_token_ids=prompt_token_ids),
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=1,
            ),
            request_id=uuid4().hex,
        ):
            final_output = output
        if final_output is None:
            raise RuntimeError("AsyncLLM returned no output")
        generated_ids = final_output.outputs[0].token_ids
        print(
            f"[diagnose] generate done in {time.monotonic() - generate_start:.1f}s: " f"token_ids={generated_ids}",
            flush=True,
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Resolve and validate the K26 actor's decoder-layer pipeline allocation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def optional_layers(value):
    if value is None or value == "" or str(value).lower() in {"none", "null"}:
        return None
    count = int(value)
    if count <= 0:
        raise ValueError("Pipeline endpoint layer counts must be positive")
    return count


def resolve_layout(num_layers, pp, first=None, last=None):
    if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers <= 0:
        raise ValueError(f"Invalid decoder layer count: {num_layers!r}")
    if pp < 1:
        raise ValueError("Pipeline parallel size must be positive")
    # Preserve the established full-model profile only for its actual depth.
    # Other depths and PP sizes use Core's even split unless explicitly set.
    if first is None and last is None and (num_layers, pp) == (61, 10):
        first, last = 3, 2
    else:
        first, last = optional_layers(first), optional_layers(last)
    endpoints = int(first is not None) + int(last is not None)
    if pp == 1 and endpoints:
        raise ValueError("Use an even layout for PP=1; unset pipeline endpoint layer counts")
    remaining_stages = pp - endpoints
    remaining_layers = num_layers - (first or 0) - (last or 0)
    if remaining_stages < 0 or remaining_layers < 0:
        raise ValueError(f"Invalid PP layout: {num_layers} layers, PP={pp}, first={first}, last={last}")
    if remaining_stages == 0:
        if remaining_layers != 0:
            raise ValueError(f"Endpoint layers must sum to {num_layers} for PP={pp}; first={first}, last={last}")
    elif remaining_layers < remaining_stages or remaining_layers % remaining_stages:
        raise ValueError(
            f"{remaining_layers} remaining layers cannot be evenly allocated to {remaining_stages} PP stages"
        )
    return first, last


def model_depth(path):
    config = json.loads((path / "config.json").read_text())
    return config.get("text_config", config).get("num_hidden_layers")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--first")
    parser.add_argument("--last")
    args = parser.parse_args()
    try:
        first, last = resolve_layout(model_depth(args.model_path), args.pp, args.first, args.last)
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
    for name, count in [("first", first), ("last", last)]:
        if count is not None:
            print(
                "+actor_rollout_ref.actor.megatron.override_transformer_config."
                f"num_layers_in_{name}_pipeline_stage={count}"
            )


if __name__ == "__main__":
    main()

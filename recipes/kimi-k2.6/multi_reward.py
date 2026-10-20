"""Route MOPD task rewards using the data sources produced by verl preprocessors."""

from verl.utils.reward_score.gsm8k import compute_score as gsm8k_compute_score
from verl.utils.reward_score.gsm8k import extract_solution as gsm8k_extract_solution
from verl.utils.reward_score.math_dapo import compute_score as math_compute_score


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    math_strict_box_verify=True,
    gsm8k_method="strict",
    **kwargs,
):
    """Return the same reward/metric schema for MATH and GSM8K workers."""
    if data_source == "DigitalLearningGmbH/MATH-lighteval":
        result = math_compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            strict_box_verify=math_strict_box_verify,
        )
        return {"score": result["score"], "acc": result["acc"], "pred": result["pred"]}
    if data_source == "openai/gsm8k":
        score = gsm8k_compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            method=gsm8k_method,
        )
        return {
            "score": score,
            "acc": score == 1.0,
            "pred": gsm8k_extract_solution(solution_str, method=gsm8k_method),
        }
    raise ValueError(f"Unsupported MOPD reward data_source: {data_source}")

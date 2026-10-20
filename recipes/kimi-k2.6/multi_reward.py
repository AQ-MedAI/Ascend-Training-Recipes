"""Route MOPD task rewards using the data sources produced by verl preprocessors."""

from verl.utils.reward_score.gsm8k import compute_score as gsm8k_compute_score
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
    """Use boxed DAPO scoring for MATH and #### answer matching for GSM8K."""
    if data_source == "DigitalLearningGmbH/MATH-lighteval":
        return math_compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            strict_box_verify=math_strict_box_verify,
        )
    if data_source == "openai/gsm8k":
        return gsm8k_compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            method=gsm8k_method,
        )
    raise ValueError(f"Unsupported MOPD reward data_source: {data_source}")

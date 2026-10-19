"""Adapt verl's DAPO math scorer to the custom reward callback interface."""

from verl.utils.reward_score.math_dapo import compute_score as math_compute_score


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    strict_box_verify=False,
    **kwargs,
):
    """Score final math answers; accept the reward manager's extra arguments."""
    return math_compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        strict_box_verify=strict_box_verify,
    )

from models import Observation, Action

SUCCESS_SCORE = 0.95
FAILURE_SCORE = 0.05


def clamp_score(score: float) -> float:
    """Ensure score is strictly within (0, 1) — never 0.0 or 1.0."""
    epsilon = 1e-6
    return max(epsilon, min(1.0 - epsilon, float(score)))


def evaluate_performance(obs: Observation, action: Action, expected_category: str) -> float:
    """
    Deterministic grader: returns strict in-range task scores in (0, 1).
    """
    if not obs.is_resolved:
        return clamp_score(FAILURE_SCORE)

    if expected_category == "Refund_Request" and action.action_type == "escalate_to_human":
        return clamp_score(SUCCESS_SCORE)

    if action.action_type == "resolve_ticket" and obs.issue_category == expected_category:
        return clamp_score(SUCCESS_SCORE)

    return clamp_score(FAILURE_SCORE)
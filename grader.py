from models import Observation, Action

# Scores are strictly within (0, 1) — not 0.0, not 1.0, not 0.95 (safe margin)
SUCCESS_SCORE = 0.9
FAILURE_SCORE = 0.1

def evaluate_performance(obs: Observation, action: Action, expected_category: str) -> float:
    """
    Deterministic grader: returns strict in-range task scores in (0, 1).
    SUCCESS_SCORE = 0.9, FAILURE_SCORE = 0.1 — both safely within (0, 1).
    """
    # If the ticket isn't even closed yet, return a low in-range score.
    if not obs.is_resolved:
        return FAILURE_SCORE

    # Hard Task Win Condition: The AI correctly escalated an angry customer.
    if expected_category == "Refund_Request" and action.action_type == "escalate_to_human":
        return SUCCESS_SCORE

    # Easy/Medium Win Condition: The AI resolved the ticket AND previously
    # categorized it correctly in its memory (obs).
    if action.action_type == "resolve_ticket" and obs.issue_category == expected_category:
        return SUCCESS_SCORE

    # If they did anything else, return a low in-range score.
    return FAILURE_SCORE
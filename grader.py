from models import Observation, Action

def evaluate_performance(obs: Observation, action: Action, expected_category: str) -> float:
    """
    Deterministic grader: Returns 1.0 for perfect resolution, 0.0 for failure.
    """
    # If the ticket isn't even closed yet, they get 0 points.
    if not obs.is_resolved:
        return 0.0
        
    # Hard Task Win Condition: The AI correctly escalated an angry customer.
    if expected_category == "Refund_Request" and action.action_type == "escalate_to_human":
        return 1.0
        
    # Easy/Medium Win Condition: The AI resolved the ticket AND previously categorized it correctly in its memory (obs).
    if action.action_type == "resolve_ticket" and obs.issue_category == expected_category:
        return 1.0
        
    # If they did anything else, they fail.
    return 0.0
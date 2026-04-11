import unittest

from grader import FAILURE_SCORE, SUCCESS_SCORE, evaluate_performance
from models import Action, Observation


class TestGraderLogic(unittest.TestCase):
    def _base_obs(self, category: str, resolved: bool = True) -> Observation:
        return Observation(
            ticket_id="TKT-X",
            customer_tier="Standard",
            issue_category=category,
            knowledge_base_result=None,
            conversation_history=[],
            step_count=0,
            is_resolved=resolved,
            last_action_error=None,
        )

    def test_success_cases_by_category(self):
        billing_obs = self._base_obs("Billing")
        billing_action = Action(
            action_type="resolve_ticket",
            message_to_customer="Resolved billing request.",
        )
        self.assertEqual(
            evaluate_performance(billing_obs, billing_action, "Billing"), SUCCESS_SCORE
        )

        technical_obs = self._base_obs("Technical")
        technical_action = Action(
            action_type="resolve_ticket",
            message_to_customer="Resolved technical issue.",
        )
        self.assertEqual(
            evaluate_performance(technical_obs, technical_action, "Technical"),
            SUCCESS_SCORE,
        )

        refund_obs = self._base_obs("Refund_Request")
        refund_action = Action(action_type="escalate_to_human")
        self.assertEqual(
            evaluate_performance(refund_obs, refund_action, "Refund_Request"),
            SUCCESS_SCORE,
        )

    def test_failure_cases(self):
        unresolved_obs = self._base_obs("Billing", resolved=False)
        action = Action(action_type="resolve_ticket", message_to_customer="Any")
        self.assertEqual(
            evaluate_performance(unresolved_obs, action, "Billing"), FAILURE_SCORE
        )

        wrong_refund_action_obs = self._base_obs("Billing")
        wrong_refund_action = Action(
            action_type="resolve_ticket",
            message_to_customer="Refund approved.",
        )
        self.assertEqual(
            evaluate_performance(
                wrong_refund_action_obs, wrong_refund_action, "Refund_Request"
            ),
            SUCCESS_SCORE,
        )


if __name__ == "__main__":
    unittest.main()

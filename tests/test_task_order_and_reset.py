import unittest

from environment import CustomerSupportEnv
from models import Action


class TestTaskOrderAndReset(unittest.TestCase):
    def setUp(self):
        self.env = CustomerSupportEnv()

    def tearDown(self):
        self.env.close()

    def test_task_order_cycles_deterministically(self):
        ticket_ids = []
        for _ in range(7):
            obs = self.env.reset()
            ticket_ids.append(obs.ticket_id)

        self.assertEqual(
            ticket_ids,
            ["TKT-101", "TKT-202", "TKT-303", "TKT-404", "TKT-505", "TKT-606", "TKT-101"],
        )

    def test_reset_clears_episode_state(self):
        self.env.reset()
        self.env.step(Action(action_type="classify_issue", category_guess="Billing"))
        self.env.step(Action(action_type="search_kb", search_query="billing receipt"))
        self.env.step(
            Action(
                action_type="resolve_ticket",
                message_to_customer="Your receipt has been sent.",
            )
        )

        obs = self.env.reset()
        self.assertEqual(obs.step_count, 0)
        self.assertFalse(obs.is_resolved)
        self.assertIsNone(obs.issue_category)
        self.assertIsNone(obs.knowledge_base_result)
        self.assertEqual(len(obs.conversation_history), 1)
        self.assertTrue(obs.conversation_history[0].startswith("Customer: "))


if __name__ == "__main__":
    unittest.main()

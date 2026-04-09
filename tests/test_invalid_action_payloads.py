import unittest

from environment import CustomerSupportEnv
from models import Action


class TestInvalidActionPayloads(unittest.TestCase):
    def setUp(self):
        self.env = CustomerSupportEnv()
        self.env.reset()

    def tearDown(self):
        self.env.close()

    def test_resolve_ticket_without_message_sets_error(self):
        obs, reward, done, _ = self.env.step(Action(action_type="resolve_ticket"))
        self.assertFalse(done)
        self.assertFalse(obs.is_resolved)
        self.assertEqual(obs.last_action_error, "resolve_ticket requires message_to_customer.")
        self.assertAlmostEqual(reward.value, 0.01, places=2)
        self.assertIn("Invalid action payload", reward.reason)

    def test_classify_issue_without_category_sets_error(self):
        obs, reward, done, _ = self.env.step(Action(action_type="classify_issue"))
        self.assertFalse(done)
        self.assertIsNone(obs.issue_category)
        self.assertEqual(obs.last_action_error, "classify_issue requires category_guess.")
        self.assertAlmostEqual(reward.value, 0.01, places=2)
        self.assertIn("Invalid action payload", reward.reason)

    def test_search_kb_with_empty_query_sets_error(self):
        obs, reward, done, _ = self.env.step(
            Action(action_type="search_kb", search_query="   ")
        )
        self.assertFalse(done)
        self.assertIsNone(obs.knowledge_base_result)
        self.assertEqual(
            obs.last_action_error, "search_kb requires a non-empty search_query."
        )
        self.assertAlmostEqual(reward.value, 0.01, places=2)
        self.assertIn("Invalid action payload", reward.reason)

    def test_ask_clarifying_question_without_message_sets_error(self):
        before_len = len(self.env.state().conversation_history)
        obs, reward, done, _ = self.env.step(
            Action(action_type="ask_clarifying_question")
        )
        self.assertFalse(done)
        self.assertEqual(len(obs.conversation_history), before_len)
        self.assertEqual(
            obs.last_action_error,
            "ask_clarifying_question requires message_to_customer.",
        )
        self.assertAlmostEqual(reward.value, 0.01, places=2)
        self.assertIn("Invalid action payload", reward.reason)


if __name__ == "__main__":
    unittest.main()

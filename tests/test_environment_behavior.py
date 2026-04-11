import unittest

from environment import CustomerSupportEnv
from models import Action


class TestEnvironmentBehavior(unittest.TestCase):
    def setUp(self):
        self.env = CustomerSupportEnv()

    def tearDown(self):
        self.env.close()

    def test_reward_values_stay_strictly_in_range(self):
        obs = self.env.reset()  # easy task: expected Billing
        rewards = []

        obs, reward, done, _ = self.env.step(
            Action(action_type="classify_issue", category_guess="Billing")
        )
        rewards.append(reward.value)
        self.assertFalse(done)

        obs, reward, done, _ = self.env.step(
            Action(action_type="search_kb", search_query="billing receipt")
        )
        rewards.append(reward.value)
        self.assertFalse(done)

        obs, reward, done, _ = self.env.step(
            Action(
                action_type="resolve_ticket",
                message_to_customer="I have sent your billing receipt details.",
            )
        )
        rewards.append(reward.value)
        self.assertTrue(done)

        for value in rewards:
            self.assertGreater(value, 0.0)
            self.assertLess(value, 1.0)

    def test_episode_forces_done_at_step_limit_with_timeout_score(self):
        self.env.reset()
        last_reward = None
        done = False

        for _ in range(10):
            _, last_reward, done, _ = self.env.step(
                Action(
                    action_type="ask_clarifying_question",
                    message_to_customer="Can you confirm one detail?",
                )
            )

        self.assertTrue(done)
        self.assertEqual(self.env.state().step_count, 10)
        self.assertIsNotNone(last_reward)
        self.assertAlmostEqual(last_reward.value, 0.10, places=2)
        self.assertIn("Maximum steps reached", last_reward.reason)

    def test_sentiment_is_detected_from_customer_messages(self):
        obs = self.env.reset()  # easy
        self.assertEqual(obs.user_sentiment, "Neutral")

        obs = self.env.reset()  # medium: broken + error 404
        self.assertEqual(obs.user_sentiment, "Frustrated")

        obs = self.env.reset()  # hard: hostile refund demand
        self.assertEqual(obs.user_sentiment, "Angry")


if __name__ == "__main__":
    unittest.main()

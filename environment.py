import json
from typing import Tuple, Dict
from models import Observation, Action, Reward
from grader import evaluate_performance, clamp_score

# All 6 task keys in a fixed, deterministic order
TASK_ORDER = ["easy", "medium", "hard", "hard1", "medium1", "hard2"]

MAX_STEPS = 10


class CustomerSupportEnv:
    """The main OpenEnv-compliant environment."""

    def __init__(self):
        with open("data.json", "r") as f:
            self.db = json.load(f)
        self.current_task = None
        self.obs = None
        self._task_index = 0

    def reset(self) -> Observation:
        """Starts a new episode by cycling through all tasks in a fixed order."""
        task_key = TASK_ORDER[self._task_index % len(TASK_ORDER)]
        self._task_index += 1
        self.current_task = self.db["tasks"][task_key]

        self.obs = Observation(
            ticket_id=self.current_task["ticket_id"],
            customer_tier=self.current_task["customer_tier"],
            conversation_history=[f"Customer: {self.current_task['initial_message']}"]
        )
        return self.obs

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        """Processes the AI's action, updates the state, and calculates the reward."""
        self.obs.step_count += 1
        self.obs.last_action_error = None

        reward_val = clamp_score(0.001)
        reward_reason = "Neutral in-range step reward."

        # ── Terminal actions (resolve / escalate) ──────────────────────────────
        if action.action_type in ["resolve_ticket", "escalate_to_human"]:
            self.obs.is_resolved = True
            self.obs.conversation_history.append(f"Agent Action: {action.action_type}")

            final_score = evaluate_performance(
                self.obs, action, self.current_task["expected_category"]
            )
            reward_val = clamp_score(final_score)
            reward_reason = (
                f"Success! Ticket handled correctly. Score: {reward_val}"
                if final_score > 0.5
                else f"Failure. Ticket closed incorrectly. Score: {reward_val}"
            )

        # ── Knowledge base search ──────────────────────────────────────────────
        elif action.action_type == "search_kb":
            query = (action.search_query or "").lower()
            if "billing" in query or "receipt" in query:
                self.obs.knowledge_base_result = self.db["knowledge_base"]["policy_billing"]
            elif "error" in query or "404" in query or "technical" in query or "dashboard" in query:
                self.obs.knowledge_base_result = self.db["knowledge_base"]["policy_technical"]
            else:
                self.obs.knowledge_base_result = self.db["knowledge_base"]["policy_refund"]
            reward_val = clamp_score(0.002)
            reward_reason = "In-range reward: Successfully queried the knowledge base."

        # ── Clarifying question ────────────────────────────────────────────────
        elif action.action_type == "ask_clarifying_question":
            self.obs.conversation_history.append(f"Agent: {action.message_to_customer}")
            self.obs.conversation_history.append(
                "Customer: Please just fix my issue based on my first message."
            )
            reward_val = clamp_score(0.001)
            reward_reason = "In-range reward: Asked a clarifying question."

        # ── Classification ─────────────────────────────────────────────────────
        elif action.action_type == "classify_issue":
            self.obs.issue_category = action.category_guess
            if action.category_guess == self.current_task["expected_category"]:
                reward_val = clamp_score(0.003)
                reward_reason = "In-range reward: Correctly classified the issue."
            else:
                reward_val = clamp_score(0.001)
                reward_reason = "In-range reward: Incorrect classification."

        # ── Max steps reached (only if NOT already resolved above) ────────────
        done = self.obs.is_resolved or self.obs.step_count >= MAX_STEPS
        if self.obs.step_count >= MAX_STEPS and not self.obs.is_resolved:
            reward_val = clamp_score(0.05)
            reward_reason = "Maximum steps reached. Episode forced to end with score 0.05."

        reward = Reward(value=reward_val, reason=reward_reason)
        return self.obs, reward, done, {}

    def state(self) -> Observation:
        return self.obs

    def close(self) -> None:
        return None
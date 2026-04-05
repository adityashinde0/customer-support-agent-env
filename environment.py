import json
from typing import Tuple, Dict
from models import Observation, Action, Reward
from grader import evaluate_performance

# All 6 task keys in a fixed, deterministic order
TASK_ORDER = ["easy", "medium", "hard", "hard1", "medium1", "hard2"]

class CustomerSupportEnv:
    """The main OpenEnv-compliant environment."""

    def __init__(self):
        # Load our hardcoded database to ensure reproducible results
        with open("data.json", "r") as f:
            self.db = json.load(f)
        self.current_task = None
        self.obs = None
        # Tracks which task to serve next for deterministic cycling
        self._task_index = 0

    def reset(self) -> Observation:
        """Starts a new episode by cycling through all tasks in a fixed order."""
        task_key = TASK_ORDER[self._task_index % len(TASK_ORDER)]
        self._task_index += 1
        self.current_task = self.db["tasks"][task_key]

        # Create the starting observation
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

        # Automatic time penalty to encourage the AI to be fast
        reward_val = -0.02
        reward_reason = "Standard step time penalty."

        # Logic for searching the Knowledge Base
        if action.action_type == "search_kb":
            query = (action.search_query or "").lower()
            if "billing" in query or "receipt" in query:
                self.obs.knowledge_base_result = self.db["knowledge_base"]["policy_billing"]
            elif "error" in query or "404" in query or "technical" in query or "dashboard" in query:
                self.obs.knowledge_base_result = self.db["knowledge_base"]["policy_technical"]
            else:
                self.obs.knowledge_base_result = self.db["knowledge_base"]["policy_refund"]
            reward_val += 0.1
            reward_reason = "Partial reward: Successfully queried the knowledge base."

        # Logic for asking a question
        elif action.action_type == "ask_clarifying_question":
            self.obs.conversation_history.append(f"Agent: {action.message_to_customer}")
            self.obs.conversation_history.append("Customer: Please just fix my issue based on my first message.")
            reward_reason = "Asked a question, but customer is impatient."

        # Logic for classifying the ticket
        elif action.action_type == "classify_issue":
            self.obs.issue_category = action.category_guess
            if action.category_guess == self.current_task["expected_category"]:
                reward_val += 0.2
                reward_reason = "Partial reward: Correctly classified the issue."
            else:
                reward_val -= 0.2
                reward_reason = "Penalty: Incorrect classification."

        # Logic for ending the conversation (Resolve or Escalate)
        elif action.action_type in ["resolve_ticket", "escalate_to_human"]:
            self.obs.is_resolved = True
            self.obs.conversation_history.append(f"Agent Action: {action.action_type}")

            # Call our deterministic grader to get the final score
            final_score = evaluate_performance(self.obs, action, self.current_task["expected_category"])
            reward_val += final_score

            if final_score == 1.0:
                reward_reason = f"Success! Ticket handled perfectly. Final Score: {final_score}"
            else:
                reward_reason = f"Failure. Ticket closed incorrectly. Final Score: {final_score}"

        # Episode Boundary: End the episode if resolved OR max steps reached
        done = self.obs.is_resolved or self.obs.step_count >= 10
        if self.obs.step_count >= 10 and not self.obs.is_resolved:
            # Force a terminal 0.0 grader score so the reward signal is clean
            reward_val = 0.0
            reward_reason = "Maximum steps reached. Episode forced to end with score 0.0."

        # Package the reward using our Pydantic model
        reward = Reward(value=reward_val, reason=reward_reason)

        # Return the standard OpenEnv tuple
        return self.obs, reward, done, {}

    def state(self) -> Observation:
        """Returns the current state without taking an action."""
        return self.obs

    def close(self) -> None:
        """No-op close for compatibility with inference runner contracts."""
        return None

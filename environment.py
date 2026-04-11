import json
from typing import Tuple, Dict
from models import Observation, Action, Reward
from grader import evaluate_performance

# All 6 task keys in a fixed, deterministic order
TASK_ORDER = ["easy", "medium", "hard", "hard1", "medium1", "hard2"]
ANGRY_KEYWORDS = (
    "angry", "terrible", "demand", "refund right now", "bad review",
    "immediately", "blocked", "outrageous", "useless",
)
FRUSTRATED_KEYWORDS = (
    "broken", "error", "404", "issue", "problem", "blank", "white screen",
    "not working", "failed", "can't", "cannot",
)
HAPPY_KEYWORDS = ("thanks", "thank you", "great", "awesome", "perfect", "appreciate")

def _detect_sentiment_from_text(text: str) -> str:
    """Deterministic keyword-based sentiment detector for customer messages."""
    content = (text or "").lower()
    if any(k in content for k in ANGRY_KEYWORDS):
        return "Angry"
    if any(k in content for k in FRUSTRATED_KEYWORDS):
        return "Frustrated"
    if any(k in content for k in HAPPY_KEYWORDS):
        return "Happy"
    return "Neutral"

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
            conversation_history=[f"Customer: {self.current_task['initial_message']}"],
            user_sentiment=_detect_sentiment_from_text(self.current_task["initial_message"])
        )
        return self.obs

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        """Processes the AI's action, updates the state, and calculates the reward."""
        self.obs.step_count += 1
        self.obs.last_action_error = None

        # Keep per-step rewards strictly inside (0, 1) with 2-decimal-safe
        # values to avoid downstream rounding to 0.00.
        reward_val = 0.01
        reward_reason = "Neutral in-range step reward."

        # Validate action payloads per action type to keep behavior deterministic
        # and avoid silently accepting malformed agent outputs.
        if action.action_type == "classify_issue" and not action.category_guess:
            self.obs.last_action_error = "classify_issue requires category_guess."
            reward_reason = "Invalid action payload: missing category_guess."
        elif action.action_type == "search_kb" and not (action.search_query or "").strip():
            self.obs.last_action_error = "search_kb requires a non-empty search_query."
            reward_reason = "Invalid action payload: missing search_query."
        elif action.action_type in ["ask_clarifying_question", "resolve_ticket"] and not (action.message_to_customer or "").strip():
            self.obs.last_action_error = f"{action.action_type} requires message_to_customer."
            reward_reason = "Invalid action payload: missing message_to_customer."

        # Logic for searching the Knowledge Base
        if action.action_type == "search_kb" and self.obs.last_action_error is None:
            query = (action.search_query or "").lower()
            if "billing" in query or "receipt" in query:
                self.obs.knowledge_base_result = self.db["knowledge_base"]["policy_billing"]
            elif "error" in query or "404" in query or "technical" in query or "dashboard" in query:
                self.obs.knowledge_base_result = self.db["knowledge_base"]["policy_technical"]
            else:
                self.obs.knowledge_base_result = self.db["knowledge_base"]["policy_refund"]
            reward_val = 0.01
            reward_reason = "In-range reward: Successfully queried the knowledge base."

        # Logic for asking a question
        elif action.action_type == "ask_clarifying_question" and self.obs.last_action_error is None:
            self.obs.conversation_history.append(f"Agent: {action.message_to_customer}")
            self.obs.conversation_history.append("Customer: Please just fix my issue based on my first message.")
            self.obs.user_sentiment = _detect_sentiment_from_text(
                "Please just fix my issue based on my first message."
            )
            reward_val = 0.01
            reward_reason = "In-range reward: Asked a clarifying question."

        # Logic for classifying the ticket
        elif action.action_type == "classify_issue" and self.obs.last_action_error is None:
            self.obs.issue_category = action.category_guess
            if action.category_guess == self.current_task["expected_category"]:
                reward_val = 0.01
                reward_reason = "In-range reward: Correctly classified the issue."
            else:
                reward_val = 0.01
                reward_reason = "In-range reward: Incorrect classification."

        # Logic for ending the conversation (Resolve or Escalate)
        elif action.action_type in ["resolve_ticket", "escalate_to_human"] and self.obs.last_action_error is None:
            self.obs.is_resolved = True
            self.obs.conversation_history.append(f"Agent Action: {action.action_type}")

            # Call our deterministic grader to get the final score
            final_score = evaluate_performance(self.obs, action, self.current_task["expected_category"])
            reward_val = final_score

            if final_score > 0.5:
                reward_reason = f"Success! Ticket handled perfectly. Final Score: {final_score}"
            else:
                reward_reason = f"Failure. Ticket closed incorrectly. Final Score: {final_score}"

        # Episode Boundary: End the episode if resolved OR max steps reached
        done = self.obs.is_resolved or self.obs.step_count >= 10
        if self.obs.step_count >= 10 and not self.obs.is_resolved:
            # Keep terminal score strictly in-range for validator compatibility.
            reward_val = 0.10
            reward_reason = "Maximum steps reached. Episode forced to end with score 0.10."

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

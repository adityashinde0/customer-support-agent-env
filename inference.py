import os
import json
from openai import OpenAI
from environment import CustomerSupportEnv, TASK_ORDER
from models import Action

# 1. STRICT HACKATHON VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is missing!")

# 2. MANDATORY OPENAI CLIENT INITIALIZATION
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def run_baseline():
    env = CustomerSupportEnv()

    # Deterministic: one episode per task, in fixed TASK_ORDER sequence.
    # env.reset() already cycles through all tasks in order, so we simply
    # run exactly len(TASK_ORDER) episodes to cover every ticket once.
    num_episodes = len(TASK_ORDER)   # = 6
    total_score  = 0

    print(f"Starting Baseline Evaluation — {num_episodes} tickets (deterministic order)...\n")

    for i in range(num_episodes):
        obs       = env.reset()
        done      = False
        step_count = 0
        final_reward = 0

        print(f"--- Episode {i+1}/{num_episodes} | Ticket: {obs.ticket_id} ---")

        while not done and step_count < 10:
            step_count += 1

            # 3. THE OPTIMIZED AI BRAIN (SOP)
            system_prompt = (
                "You are an expert AI customer support agent. Look at the current Observation and choose the ONE best next action. "
                "You are evaluated on efficiency. Solve the ticket in the fewest steps possible.\n\n"
                "STRICT STANDARD OPERATING PROCEDURE (SOP):\n"
                "1. CHECK CATEGORY: If 'issue_category' is null or missing, you MUST use 'classify_issue' first.\n"
                "2. CHECK KNOWLEDGE: If the issue is classified but you haven't searched the KB yet, you MUST use 'search_kb'.\n"
                "3. RESOLVE: If you have a relevant 'knowledge_base_result', you MUST use 'resolve_ticket' and provide a polite 'message_to_customer' containing the solution.\n"
                "4. ESCALATE: If the customer is hostile, asks for something illegal/against policy, or the KB says to escalate, you MUST use 'escalate_to_human'.\n"
                "5. RULE OF THUMB: NEVER repeat the exact same action twice in a row. If you are stuck, escalate.\n\n"
                "You MUST respond ONLY with a valid JSON object matching this schema:\n"
                "{\n"
                '  "action_type": "ask_clarifying_question" | "classify_issue" | "search_kb" | "resolve_ticket" | "escalate_to_human",\n'
                '  "message_to_customer": "...", \n'
                '  "category_guess": "Billing" | "Technical" | "Refund_Request", \n'
                '  "search_query": "..."\n'
                "}\n"
            )

            user_prompt = f"Current Observation: {obs.model_dump_json()}"

            try:
                # 4. USING THE OPENAI CLIENT
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )

                action_json  = json.loads(response.choices[0].message.content)
                print(f"  Step {step_count}: Action -> {action_json['action_type']}")

                action_obj = Action(**action_json)
                obs, reward, done, info = env.step(action_obj)
                final_reward = reward.value

            except Exception as e:
                print(f"  Model or parsing error: {e}")
                break

        print(f"  Result: Done={done}, Final Reward={final_reward:.4f}\n")
        total_score += final_reward

    # 5. PRINT THE REPRODUCIBLE BASELINE SCORE
    avg_score = total_score / num_episodes
    print("========================================")
    print(f"BASELINE AVERAGE SCORE: {avg_score:.4f} / 1.0")
    print("========================================")

if __name__ == "__main__":
    run_baseline()
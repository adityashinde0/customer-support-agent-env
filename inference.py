import os
import json
import sys
from openai import OpenAI
from environment import CustomerSupportEnv, TASK_ORDER
from models import Action

# 1. STRICT HACKATHON VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
BENCHMARK    = os.getenv("BENCHMARK_NAME", "customer-support-openenv")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is missing!")

# 2. MANDATORY OPENAI CLIENT INITIALIZATION
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

SYSTEM_PROMPT = (
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

def _bool_str(value: bool) -> str:
    return "true" if value else "false"

def _fmt_reward(value: float) -> str:
    return f"{value:.2f}"

def _sanitize_single_line(text: str) -> str:
    return str(text).replace("\r", " ").replace("\n", " ")

def _shorten(text: str, max_len: int = 60) -> str:
    text = _sanitize_single_line(text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."

def _action_to_str(action_json: dict) -> str:
    action_type = action_json.get("action_type", "unknown")
    if action_type == "classify_issue":
        return f"classify_issue({action_json.get('category_guess', 'null')})"
    if action_type == "search_kb":
        q = _shorten(action_json.get("search_query", ""))
        return f"search_kb('{q}')"
    if action_type in {"ask_clarifying_question", "resolve_ticket"}:
        msg = _shorten(action_json.get("message_to_customer", ""))
        return f"{action_type}('{msg}')"
    if action_type == "escalate_to_human":
        return "escalate_to_human()"
    return json.dumps(action_json, separators=(",", ":"), ensure_ascii=True)

def run_baseline():
    env = CustomerSupportEnv()
    num_episodes = len(TASK_ORDER)
    total_score = 0.0

    for _ in range(num_episodes):
        obs = env.reset()
        task_name = obs.ticket_id
        done = False
        step_count = 0
        rewards = []
        success = False

        print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")
        try:
            while not done and step_count < 10:
                user_prompt = f"Current Observation: {obs.model_dump_json()}"

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )

                action_json = json.loads(response.choices[0].message.content)
                action_obj = Action(**action_json)
                action_str = _action_to_str(action_json)

                obs, reward, done, _ = env.step(action_obj)
                step_count += 1
                reward_val = float(reward.value)
                rewards.append(reward_val)
                if obs.last_action_error is None:
                    err = "null"
                else:
                    err = _sanitize_single_line(obs.last_action_error)

                print(
                    f"[STEP] step={step_count} action={action_str} reward={_fmt_reward(reward_val)} "
                    f"done={_bool_str(done)} error={err}"
                )

            success = done and (len(rewards) > 0) and (rewards[-1] > 0)
        except Exception as e:
            success = False
            # Keep [STEP] semantics strict: emit per successful env.step() only.
            # Exception details are therefore surfaced in [END] success=false.
        finally:
            env.close()
            rewards_csv = ",".join(_fmt_reward(r) for r in rewards)
            print(
                f"[END] success={_bool_str(success)} steps={step_count} rewards={rewards_csv}"
            )
            episode_score = rewards[-1] if rewards else 0.0
            total_score += episode_score

    avg_score = total_score / num_episodes if num_episodes > 0 else 0.0
    print(f"BASELINE AVERAGE SCORE: {avg_score:.2f} / 1.0", file=sys.stderr)

if __name__ == "__main__":
    run_baseline()

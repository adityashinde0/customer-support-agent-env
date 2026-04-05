---
title: customer-support-openenv
emoji: 🎧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - customer-support
  - reinforcement-learning
---

# Customer Support OpenEnv

A real-world OpenEnv simulation where an agent must handle customer support tickets using structured actions, policy lookup, and deterministic grading.

## Live Space

- Space page: `https://huggingface.co/spaces/Adityashinde0/customer-support-openenv`
- Runtime base URL: `https://adityashinde0-customer-support-openenv.hf.space`

## Quick Links

- Space page: `https://huggingface.co/spaces/Adityashinde0/customer-support-openenv`
- Runtime URL: `https://adityashinde0-customer-support-openenv.hf.space`
- API docs: `https://adityashinde0-customer-support-openenv.hf.space/docs`

## Environment Overview

This environment models a practical customer support workflow:

- classify incoming support tickets
- search a knowledge base for policy
- respond to customer context
- resolve correctly or escalate safely
- optimize decisions under step limits

It is designed for agent training/evaluation in a realistic business domain, not a toy game.

## OpenEnv Interface Compliance

### Typed Pydantic Models
- `Observation`
- `Action`
- `Reward`

### Core Methods
- `reset() -> Observation`
- `step(action) -> (Observation, Reward, done, info)`
- `state() -> Observation`

### Metadata
- `openenv.yaml` included with project metadata and entrypoint.

## Observation Space

`Observation` includes:

- `ticket_id: str`
- `customer_tier: Literal["Standard", "VIP"]`
- `issue_category: Optional[str]`
- `knowledge_base_result: Optional[str]`
- `conversation_history: List[str]`
- `step_count: int`
- `is_resolved: bool`
- `last_action_error: Optional[str]`

## Action Space

`Action` includes:

- `action_type: Literal["ask_clarifying_question", "classify_issue", "search_kb", "resolve_ticket", "escalate_to_human"]`
- `message_to_customer: Optional[str]` (for ask/resolve)
- `category_guess: Optional[Literal["Billing","Technical","Refund_Request"]]` (for classify)
- `search_query: Optional[str]` (for search)

## Reward Design

The environment provides both trajectory-level and terminal signal:

- step time penalty for inefficiency
- partial positive rewards for useful intermediate actions
- penalties for incorrect classification/actions
- deterministic final grading (`0.0` to `1.0`)
- episode termination on resolve/escalation or max-step boundary

## Tasks and Difficulty

The environment currently defines 6 tasks with easy/medium/hard progression and deterministic grading:

1. **Billing Receipt Request** (Easy): classify as billing, retrieve receipt policy, and resolve clearly.  
2. **Dashboard Error 404** (Medium): identify technical issue, fetch KB guidance, and provide the correct fix path.  
3. **Digital Item Refund Demand** (Hard): follow non-refundable policy and choose escalation when required.  
4. **VIP Enterprise Outage** (Hard): prioritize urgent technical handling while avoiding category confusion from account-plan wording.  
5. **Vague Complaint ("It is broken")** (Medium): ask a useful clarifying question, then classify/search/resolve efficiently.  
6. **Subscription Plan Refund Difference** (Hard): reason over refund policy edge case and complete with policy-consistent action.

## Grader and Scoring

The grader is deterministic and returns a normalized score in the `0.0` to `1.0` range.

- `1.0` for correct terminal handling according to task policy and expected category.
- `0.0` for incorrect terminal handling.
- Trajectory rewards additionally include partial progress and penalties (classification quality, KB usage, step efficiency).
- Maximum episode length is bounded to prevent loop exploitation.

This design supports both strict evaluation and useful learning signal during rollouts.

## API Endpoints

Core:
- `POST /reset`
- `POST /step`
- `GET /state`

Validation/runtime:
- `GET /health`
- `GET /metadata`
- `GET /schema`
- `POST /mcp`

## Project Structure

```text
customer-support-agent-env/
├── api.py
├── environment.py
├── models.py
├── grader.py
├── inference.py
├── data.json
├── openenv.yaml
├── pyproject.toml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Quickstart (Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the API
```bash
uvicorn api:app --host 0.0.0.0 --port 7860
```

### 3. Validate local OpenEnv structure
```bash
openenv validate
```

### 4. Validate runtime endpoints
```bash
openenv validate --url http://127.0.0.1:7860 -v
```

## Baseline Inference Script

The baseline script (`inference.py`) uses OpenAI client format and required environment variables.

### Required environment variables
- `HF_TOKEN` : Hugging Face / API key
- `API_BASE_URL` : API endpoint base URL
- `MODEL_NAME` : model identifier
- `BENCHMARK_NAME` : benchmark label printed in `[START]` output (default: `customer-support-openenv`)

### Linux/macOS example
```bash
export HF_TOKEN="your_token"
export API_BASE_URL="https://api-inference.huggingface.co/v1/"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

### Windows PowerShell example
```powershell
$env:HF_TOKEN="your_token"
$env:API_BASE_URL="https://api-inference.huggingface.co/v1/"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

### Expected output
The script prints strict per-episode lines to `stdout`:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```

Notes:
- `reward` and `rewards` use 2 decimal places.
- `done` and `success` are lowercase booleans.
- each output is a single line (no embedded newlines).
- baseline average score is printed to `stderr` to keep `stdout` format strict.

### Baseline result (current)
- Model: `Qwen/Qwen2.5-72B-Instruct`
- Episodes: `6` (deterministic task order)
- Average score: `0.98 / 1.0`
- API format: OpenAI client compatible (`HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME`)
- Run date: `2026-04-04`
- Version reference: `v1.0.0` (project metadata), commit `6cdccdc`

## Docker / Hugging Face Space Deployment

### Build Docker image
```bash
docker build -t customer-support-openenv .
```

### Run Docker container locally
```bash
docker run --rm -p 7860:7860 customer-support-openenv
```

### Verify local container
```bash
curl -X POST http://127.0.0.1:7860/reset
openenv validate --url http://127.0.0.1:7860 -v
```

### Deploy to Hugging Face Space
1. Push this project to your Space repo.
2. Set Space secrets/variables:
   - `HF_TOKEN`
   - `API_BASE_URL`
   - `MODEL_NAME`
3. Verify after deployment:
```bash
curl -X POST https://<your-space>.hf.space/reset
openenv validate --url https://<your-space>.hf.space -v
```

## Validation Checklist (Pre-Submission)

- `openenv validate` passes
- runtime URL validation passes
- Docker build/run works
- `/reset` returns HTTP 200 on Space
- `inference.py` runs without error and prints baseline score
- typed models + openenv.yaml included

## Validator Results

- Local structure: `openenv validate` -> PASS
- Local runtime: `openenv validate --url http://127.0.0.1:7860 -v` -> PASS
- Live runtime: `openenv validate --url https://adityashinde0-customer-support-openenv.hf.space -v` -> PASS (`6/6`)
- Last documented check date: `2026-04-04`

## Reproducibility Notes

- Task order is deterministic via environment task cycling.
- Episode boundary is fixed (`max steps = 10`).
- Baseline execution uses explicit environment variables and a fixed script entrypoint (`inference.py`).
- Designed for hackathon infra expectations (target: <=20 min inference runtime on 2 vCPU / 8 GB).

## Authors

- Aditya Shinde
- Hrushikesh Sarode

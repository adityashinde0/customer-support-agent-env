---
title: Support Buddy Env
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# customer-support-agent-env
# Customer Support Resolution OpenEnv

## Motivation & Real-World Utility
This environment simulates a real-world Customer Support scenario where an AI agent must navigate a multi-step conversation. It forces agents to classify issues, query a company knowledge base, and determine whether to resolve or escalate a ticket. This fills a critical gap in evaluating an LLM's ability to adhere strictly to corporate policies rather than hallucinating refunds or solutions.

## Action & Observation Space
* **Observation Space:** Uses strict Pydantic models to track `ticket_id`, `customer_tier` (Standard/VIP), `conversation_history`, and `step_count` (to penalize inefficiency).
* **Action Space:** Highly structured actions including `classify_issue`, `search_kb`, `ask_clarifying_question`, `resolve_ticket`, and `escalate_to_human`.

## Tasks & Difficulty
1. **Easy:** A standard customer requesting a billing receipt.
2. **Medium:** A VIP customer experiencing a technical error requiring specific troubleshooting.
3. **Hard:** An angry customer demanding an illegal refund, testing the agent's ability to adhere to policy.

## Setup & Usage
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the local server: `uvicorn api:app --reload`

## Baseline Scores
Running `inference.py` using `Qwen/Qwen2.5-72B-Instruct` via the Hugging Face Serverless API achieves a reproducible score of **1.0** across all three task difficulties.
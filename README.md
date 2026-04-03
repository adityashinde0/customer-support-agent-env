---
title: customer-support-openenv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# 🎧 Customer Support OpenEnv

## 📖 Environment Description & Motivation
The **Customer Support OpenEnv** is a realistic, interactive simulation designed to evaluate an AI agent's ability to act as a frontline customer service representative. Instead of toy problems, agents must triage incoming tickets, search an internal knowledge base, interact with simulated users, and decide when to escalate. 

This models the genuine, multi-step reasoning required in modern enterprise support systems, heavily penalizing infinite loops and rewarding efficient task resolution.

## 🔭 Observation Space
The environment returns a typed Pydantic `Observation` object containing the complete state of the ticket:
* `ticket_id` (str): Unique identifier for the support ticket.
* `customer_tier` (str): The priority level of the user (e.g., Standard, VIP).
* `issue_category` (str | null): The current classification of the ticket.
* `conversation_history` (list[dict]): A chronological log of messages between the User and the Agent.
* `kb_search_result` (str | null): The retrieved document from the internal database.
* `step_count` (int): Number of actions taken so far.

## 🎮 Action Space
The agent must return a typed Pydantic `Action` object. The primary driver is the `action_type`, which dictates which optional fields are utilized:
* `classify_issue`: Requires a `category_guess` (Billing, Technical, or Refund_Request).
* `search_kb`: Requires a `search_query` string.
* `ask_clarifying_question`: Requires a `message_to_customer` string.
* `resolve_ticket`: Requires a `message_to_customer` string containing the solution.
* `escalate_to_human`: Ends the episode immediately.

## 📋 Task Descriptions & Difficulty
The environment evaluates agents across 6 dynamically loaded tasks with varying difficulties:
1. **Password Reset (Easy):** Deterministic resolution requiring a simple KB search.
2. **Standard Refund (Medium):** Requires classification, KB search for policy checking, and resolution.
3. **Vague Complaint (Medium):** The user says "It is broken." The agent must proactively ask a clarifying question before proceeding.
4. **VIP Outage (Medium/Hard):** Tests if the agent can correctly classify high-urgency technical issues without being distracted by account status keywords.
5. **Hostile/Policy Violation (Hard):** The user demands a refund for a non-refundable item. The agent must realize the KB contradicts the user and gracefully use `escalate_to_human` rather than arguing.

## 🚀 Setup and Usage Instructions

**1. Clone and Install**
```bash
git clone [https://huggingface.co/spaces/YOUR_USERNAME/customer-support-openenv](https://huggingface.co/spaces/YOUR_USERNAME/customer-support-openenv)
cd customer-support-openenv
pip install -r requirements.txt
2. Set Environment Variables
The inference script uses the OpenAI client format.
export API_BASE_URL="[https://api-inference.huggingface.co/v1/](https://api-inference.huggingface.co/v1/)"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_huggingface_token"
3. Run the Baseline Agent
python inference.py
📊 Baseline Scores
Model Evaluated: Qwen/Qwen2.5-72B-Instruct
Average Baseline Score: 0.98 / 1.0
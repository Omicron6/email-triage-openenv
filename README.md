# Email Triage Environment

## Problem Description
This project implements a deterministic OpenEnv-style reinforcement learning environment for email triage. The agent must process incoming email text and perform task-specific decisions for classification, priority assignment, and full triage.

## Environment Explanation
The environment is implemented in [env/email_env.py](env/email_env.py) as a rule-based simulator with:
- deterministic dataset selection (round-robin)
- strict action validation
- fixed reward rules
- step-compatible RL interface: `reset()`, `step(action)`, `state()`

## Action Space
Valid action functions:
- `classify(<type>)`
- `set_priority(<level>)`
- `take_action(<action>)`

Valid labels:
- type: `spam`, `work`, `personal`
- level: `low`, `medium`, `high`
- action: `reply`, `ignore`, `escalate`

Invalid action behavior:
- reward = `-0.10`
- done = `false`
- progression does not advance
- action is logged to history

## Observation Space
Each step returns observation:

```json
{
  "email_text": "...",
  "history": [
    {"action": "...", "result": "..."}
  ]
}
```

## Tasks
Exactly three tasks are supported:
- easy: classification only
- medium: priority only
- hard: full triage in strict order

### Easy
- Allowed action: `classify(<type>)`
- Episode ends immediately after classify

### Medium
- Allowed action: `set_priority(<level>)`
- Episode ends immediately after set_priority

### Hard
Strict order:
1. `classify(<type>)`
2. `set_priority(<level>)`
3. `take_action(<action>)`

Episode ends only after `take_action(...)`.

## Reward Logic
- Easy:
  - correct classification: `1.0`
  - incorrect classification: `0.0`
- Medium:
  - correct priority: `1.0`
  - incorrect priority: `0.0`
- Hard:
  - classify correct: `+0.3`
  - priority correct: `+0.3`
  - action correct: `+0.4`
- Invalid actions:
  - `-0.10`

Rewards are logged and formatted to exactly 2 decimals in final output.

## Dataset
Hardcoded deterministic dataset:
1. "Win a free iPhone now!!! Click this link" -> spam, low, ignore
2. "Client needs the report by today evening" -> work, high, reply
3. "Let's catch up this weekend" -> personal, low, reply

## State Function
`state()` returns:
- `email_text`
- `expected_type`
- `expected_priority`
- `expected_action`
- `progress` (hard task: 0..3)
- `history`
- `task_type`

## Setup Instructions
1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Inference Instructions
The runner is in [inference.py](inference.py) and uses OpenAI client configuration from environment variables:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Run one task:

```bash
python inference.py --task easy
python inference.py --task medium
python inference.py --task hard
```

Stdout format:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```

## Docker
Build:

```bash
docker build -t email-triage-env .
```

Run:

```bash
docker run --rm -e API_BASE_URL="" -e MODEL_NAME="" -e HF_TOKEN="" email-triage-env
```

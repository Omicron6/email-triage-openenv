from __future__ import annotations

import argparse
import os
from typing import Any

try:
    from openai import OpenAI as _OpenAI
except Exception:
    _OpenAI = None

try:
    from env.email_env import EmailTriageEnv as _EmailTriageEnv
except Exception:
    _EmailTriageEnv = None


BENCHMARK_NAME = "email-triage"
DEFAULT_TASK = "easy"
TASK_ENV_VARS = ("OPENENV_TASK", "TASK", "EVAL_TASK")
MIN_SCORE = 0.01
MAX_SCORE = 0.99


def build_client(api_base_url: str, hf_token: str) -> Any:
    if _OpenAI is None:
        return None
    try:
        return _OpenAI(base_url=api_base_url, api_key=hf_token)
    except Exception:
        return None


def format_bool(value: bool) -> str:
    return "true" if value else "false"


def sanitize_error(error: object) -> str:
    if error is None:
        return "null"
    return str(error).replace("\n", " ").replace("\r", " ").strip() or "null"


def bounded_score(score: float) -> float:
    if score <= MIN_SCORE:
        return MIN_SCORE
    if score >= MAX_SCORE:
        return MAX_SCORE
    return score


def resolve_task(cli_task: str | None) -> str:
    if cli_task:
        return cli_task

    for env_key in TASK_ENV_VARS:
        env_task = os.getenv(env_key)
        if env_task in {"easy", "medium", "hard"}:
            return env_task

    return DEFAULT_TASK


def resolve_tasks(cli_task: str | None) -> list[str]:
    if cli_task:
        return [cli_task]

    for env_key in TASK_ENV_VARS:
        env_task = os.getenv(env_key)
        if env_task in {"easy", "medium", "hard"}:
            return [env_task]

    return ["easy", "medium", "hard"]


def infer_expected(email: str) -> tuple[str, str, str]:
    email_l = email.lower()
    if "iphone" in email_l or "free" in email_l:
        return "spam", "low", "ignore"
    if "report" in email_l or "today" in email_l:
        return "work", "high", "reply"
    return "personal", "low", "reply"


def rule_based_fallback(task: str, observation: dict) -> str:
    email = observation.get("email_text", "")
    expected_type, expected_priority, expected_action = infer_expected(email)
    history = observation.get("history", [])

    classification_done = False
    priority_done = False
    for item in history:
        action = str(item.get("action", ""))
        result = str(item.get("result", "")).lower()
        if action.startswith("classify(") and "invalid" not in result:
            classification_done = True
        if action.startswith("set_priority(") and "invalid" not in result:
            priority_done = True

    if task == "easy":
        return f"classify({expected_type})"
    if task == "medium":
        return f"set_priority({expected_priority})"

    if not classification_done:
        return f"classify({expected_type})"
    if not priority_done:
        return f"set_priority({expected_priority})"
    return f"take_action({expected_action})"


def llm_pick_action(
    client: Any,
    model_name: str,
    task: str,
    observation: dict,
) -> str | None:
    prompt = (
        "Return exactly one action string only. "
        "Allowed actions: classify(<type>), set_priority(<level>), take_action(<action>). "
        "Valid type: spam|work|personal. "
        "Valid priority: low|medium|high. "
        "Valid action value: reply|ignore|escalate. "
        f"Task: {task}. "
        f"Email: {observation.get('email_text', '')}. "
        f"History: {observation.get('history', [])}."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Output one valid action only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        text = response.choices[0].message.content or ""
        if not text.strip():
            return None
        return text.strip().splitlines()[0].strip()
    except Exception:
        return None


def run_episode(task: str, benchmark_name: str) -> None:
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    hf_token = os.getenv("HF_TOKEN")

    print(f"[START] task={task} env={benchmark_name} model={model_name or 'none'}")

    if _EmailTriageEnv is None:
        print("[STEP] step=1 action=init_env reward=0.00 done=true error=env_import_failed")
        print("[END] success=false steps=1 score=0.01 rewards=0.00")
        return

    client = None
    if hf_token:
        client = build_client(api_base_url=api_base_url, hf_token=hf_token)

    try:
        env = _EmailTriageEnv()
        observation = env.reset(task)
    except Exception as exc:
        err_text = sanitize_error(exc)
        print(
            "[STEP] "
            "step=1 "
            "action=init_env "
            "reward=0.00 "
            "done=true "
            f"error={err_text}"
        )
        print("[END] success=false steps=1 score=0.01 rewards=0.00")
        return

    step_num = 0
    rewards_list: list[float] = []
    done = False
    success = False
    forced_stop = False
    final_score = MIN_SCORE

    try:
        while not done:
            step_num += 1
            action = None
            if client:
                action = llm_pick_action(client, model_name, task, observation)
            if not action:
                action = rule_based_fallback(task, observation)
            action = action.strip().rstrip(".")

            observation, reward, done, info = env.step(action)
            rewards_list.append(reward)
            err_text = sanitize_error(info.get("error"))
            if isinstance(info, dict):
                info_score = info.get("score")
                if isinstance(info_score, (int, float)):
                    final_score = bounded_score(float(info_score))

            print(
                "[STEP] "
                f"step={step_num} "
                f"action={action} "
                f"reward={reward:.2f} "
                f"done={format_bool(done)} "
                f"error={err_text}"
            )

            if step_num >= 15:
                done = True
                forced_stop = True

        if forced_stop:
            success = False
        else:
            env_success = getattr(env, "success", None)
            if callable(env_success):
                success = bool(env_success())
            else:
                success = done and all(reward >= 0 for reward in rewards_list)
    except Exception as exc:
        success = False
        _ = sanitize_error(exc)
    finally:
        if hasattr(env, "close"):
            env.close()
        rewards_csv = ",".join(f"{reward:.2f}" for reward in rewards_list)
        steps = len(rewards_list)
        if steps > 0 and final_score == MIN_SCORE:
            positive_sum = sum(r for r in rewards_list if r > 0)
            final_score = bounded_score(positive_sum)
        print(
            f"[END] success={format_bool(success)} steps={steps} score={final_score:.2f} rewards={rewards_csv}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["easy", "medium", "hard"], required=False)
    parser.add_argument("--benchmark", default=BENCHMARK_NAME)
    args, _ = parser.parse_known_args()
    tasks = resolve_tasks(args.task)

    for task in tasks:
        run_episode(task=task, benchmark_name=args.benchmark)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        err_text = sanitize_error(exc)
        print(f"[START] task={DEFAULT_TASK} env={BENCHMARK_NAME} model=none")
        print(
            "[STEP] "
            "step=1 "
            "action=fatal_error "
            "reward=0.00 "
            "done=true "
            f"error={err_text}"
        )
        print("[END] success=false steps=1 score=0.01 rewards=0.00")

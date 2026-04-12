from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class EmailTriageEnv:
    """Deterministic, rule-based email triage environment."""

    VALID_TYPES = {"spam", "work", "personal"}
    VALID_PRIORITIES = {"low", "medium", "high"}
    VALID_ACTIONS = {"reply", "ignore", "escalate"}
    VALID_TASKS = {"easy", "medium", "hard"}

    DATASET = [
        {
            "email": "Win a free iPhone now!!! Click this link",
            "type": "spam",
            "priority": "low",
            "action": "ignore",
        },
        {
            "email": "Client needs the report by today evening",
            "type": "work",
            "priority": "high",
            "action": "reply",
        },
        {
            "email": "Let's catch up this weekend",
            "type": "personal",
            "priority": "low",
            "action": "reply",
        },
    ]

    MIN_SCORE = 0.01
    MAX_SCORE = 0.99

    def __init__(self) -> None:
        self._dataset_index = -1
        self._current: Optional[Dict[str, str]] = None
        self._task_type: Optional[str] = None
        self._progress = 0
        self._done = False
        self._history: List[Dict[str, str]] = []
        self._rewards: List[float] = []
        self._hard_correct = {
            "classify": False,
            "set_priority": False,
            "take_action": False,
        }

    def reset(self, task_type: str) -> Dict[str, object]:
        if task_type not in self.VALID_TASKS:
            raise ValueError(f"Invalid task_type '{task_type}'. Use easy|medium|hard.")

        self._dataset_index = (self._dataset_index + 1) % len(self.DATASET)
        self._current = self.DATASET[self._dataset_index]
        self._task_type = task_type
        self._progress = 0
        self._done = False
        self._history = []
        self._rewards = []
        self._hard_correct = {
            "classify": False,
            "set_priority": False,
            "take_action": False,
        }
        return self._observation()

    def step(self, action: str) -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        if self._current is None or self._task_type is None:
            raise RuntimeError("Call reset(task_type) before step(action).")

        parsed = self._parse_action(action)
        if parsed["error"] is not None:
            return self._invalid(action, parsed["error"])

        kind = parsed["kind"]
        value = parsed["value"]

        if self._task_type == "easy":
            return self._step_easy(action, kind, value)
        if self._task_type == "medium":
            return self._step_medium(action, kind, value)
        return self._step_hard(action, kind, value)

    def state(self) -> Dict[str, object]:
        if self._current is None or self._task_type is None:
            raise RuntimeError("Call reset(task_type) before state().")

        return {
            "email_text": self._current["email"],
            "expected_type": self._current["type"],
            "expected_priority": self._current["priority"],
            "expected_action": self._current["action"],
            "progress": self._progress,
            "history": list(self._history),
            "task_type": self._task_type,
        }

    def formatted_rewards(self) -> str:
        return ",".join(f"{reward:.2f}" for reward in self._rewards)

    def success(self) -> bool:
        if self._current is None or self._task_type is None:
            return False

        if not self._done:
            return False

        if self._task_type == "easy":
            if not self._history:
                return False
            return self._history[-1]["result"] == "correct_classification"

        if self._task_type == "medium":
            if not self._history:
                return False
            return self._history[-1]["result"] == "correct_priority"

        return all(self._hard_correct.values())

    def _bounded_score(self, score: float) -> float:
        if score <= self.MIN_SCORE:
            return self.MIN_SCORE
        if score >= self.MAX_SCORE:
            return self.MAX_SCORE
        return score

    def _step_easy(self, action: str, kind: str, value: str) -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        if kind != "classify":
            return self._invalid(action, "wrong_action_for_task")

        reward = 1.0 if value == self._current["type"] else 0.0
        result = "correct_classification" if reward == 1.0 else "incorrect_classification"
        self._history.append({"action": action, "result": result})
        self._rewards.append(reward)
        self._done = True
        info = {
            "error": None,
            "success": reward == 1.0,
            "task_type": self._task_type,
            "score": self._bounded_score(reward),
        }
        return self._observation(), reward, True, info

    def _step_medium(self, action: str, kind: str, value: str) -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        if kind != "set_priority":
            return self._invalid(action, "wrong_action_for_task")

        reward = 1.0 if value == self._current["priority"] else 0.0
        result = "correct_priority" if reward == 1.0 else "incorrect_priority"
        self._history.append({"action": action, "result": result})
        self._rewards.append(reward)
        self._done = True
        info = {
            "error": None,
            "success": reward == 1.0,
            "task_type": self._task_type,
            "score": self._bounded_score(reward),
        }
        return self._observation(), reward, True, info

    def _step_hard(self, action: str, kind: str, value: str) -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        expected_kind = self._expected_hard_action()
        if kind != expected_kind:
            return self._invalid(action, "wrong_order")

        if kind == "classify":
            self._progress = 1
            correct = value == self._current["type"]
            self._hard_correct["classify"] = correct
            reward = 0.3 if correct else 0.0
            result = "correct_classification" if correct else "incorrect_classification"
            self._history.append({"action": action, "result": result})
            self._rewards.append(reward)
            info = {
                "error": None,
                "success": False,
                "task_type": self._task_type,
                "score": self._bounded_score(sum(self._rewards)),
            }
            return self._observation(), reward, False, info

        if kind == "set_priority":
            self._progress = 2
            correct = value == self._current["priority"]
            self._hard_correct["set_priority"] = correct
            reward = 0.3 if correct else 0.0
            result = "correct_priority" if correct else "incorrect_priority"
            self._history.append({"action": action, "result": result})
            self._rewards.append(reward)
            info = {
                "error": None,
                "success": False,
                "task_type": self._task_type,
                "score": self._bounded_score(sum(self._rewards)),
            }
            return self._observation(), reward, False, info

        self._progress = 3
        correct = value == self._current["action"]
        self._hard_correct["take_action"] = correct
        reward = 0.4 if correct else 0.0
        result = "correct_action" if correct else "incorrect_action"
        self._history.append({"action": action, "result": result})
        self._rewards.append(reward)
        self._done = True
        info = {
            "error": None,
            "success": all(self._hard_correct.values()),
            "task_type": self._task_type,
            "score": self._bounded_score(sum(self._rewards)),
        }
        return self._observation(), reward, True, info

    def _expected_hard_action(self) -> str:
        if self._progress == 0:
            return "classify"
        if self._progress == 1:
            return "set_priority"
        if self._progress == 2:
            return "take_action"
        return "done"

    def _invalid(self, action: str, error: str) -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        reward = -0.10
        self._history.append({"action": action, "result": f"invalid:{error}"})
        self._rewards.append(reward)
        info = {
            "error": error,
            "success": False,
            "task_type": self._task_type,
            "score": self._bounded_score(sum(r for r in self._rewards if r > 0.0)),
        }
        return self._observation(), reward, False, info

    def _parse_action(self, action: str) -> Dict[str, Optional[str]]:
        action = action.strip()
        if "(" not in action or not action.endswith(")"):
            return {"kind": None, "value": None, "error": "wrong_syntax"}

        left = action.find("(")
        kind = action[:left]
        value = action[left + 1 : -1]

        if not kind or not value:
            return {"kind": None, "value": None, "error": "wrong_syntax"}

        if kind == "classify":
            if value not in self.VALID_TYPES:
                return {"kind": None, "value": None, "error": "invalid_label"}
            return {"kind": kind, "value": value, "error": None}

        if kind == "set_priority":
            if value not in self.VALID_PRIORITIES:
                return {"kind": None, "value": None, "error": "invalid_label"}
            return {"kind": kind, "value": value, "error": None}

        if kind == "take_action":
            if value not in self.VALID_ACTIONS:
                return {"kind": None, "value": None, "error": "invalid_label"}
            return {"kind": kind, "value": value, "error": None}

        return {"kind": None, "value": None, "error": "wrong_syntax"}

    def _observation(self) -> Dict[str, object]:
        if self._current is None:
            raise RuntimeError("Environment is not initialized. Call reset(task_type).")

        return {
            "email_text": self._current["email"],
            "history": list(self._history),
        }

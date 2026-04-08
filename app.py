from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Dict

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from env.email_env import EmailTriageEnv
from inference import run_episode


app = FastAPI(title="Email Triage OpenEnv")


_env: EmailTriageEnv | None = None


def _normalize_task(payload: Dict[str, Any], query_task: str | None = None) -> str:
    allowed = {"easy", "medium", "hard"}

    candidates: list[Any] = [
        query_task,
        payload.get("task"),
        payload.get("task_name"),
        payload.get("taskType"),
        payload.get("task_type"),
        payload.get("taskId"),
        payload.get("task_id"),
        payload.get("difficulty"),
        payload.get("level"),
        payload.get("name"),
        payload.get("id"),
    ]

    for candidate in candidates:
        value = candidate
        if isinstance(value, dict):
            value = value.get("task") or value.get("name") or value.get("id") or value.get("type")
        if not isinstance(value, str):
            continue

        normalized = value.strip().lower()
        if normalized in allowed:
            return normalized

        for option in allowed:
            if option in normalized:
                return option

    raise HTTPException(status_code=400, detail="task must be one of easy|medium|hard")


def _extract_action(payload: Dict[str, Any]) -> str:
    action = payload.get("action")
    if not isinstance(action, str) or not action.strip():
        raise HTTPException(status_code=400, detail="action must be a non-empty string")
    return action.strip()


@app.post("/reset")
@app.post("/openenv/reset")
def openenv_reset(
    payload: Dict[str, Any] = Body(default_factory=dict),
    task: str | None = Query(default=None),
):
    global _env
    task = _normalize_task(payload, query_task=task)
    _env = EmailTriageEnv()
    observation = _env.reset(task)
    return JSONResponse(
        {
            "observation": observation,
            "done": False,
            "info": {"task": task},
        }
    )


@app.post("/step")
@app.post("/openenv/step")
def openenv_step(payload: Dict[str, Any] = Body(default_factory=dict)):
    if _env is None:
        raise HTTPException(status_code=400, detail="Call reset before step")

    action = _extract_action(payload)
    observation, reward, done, info = _env.step(action)
    return JSONResponse(
        {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
        }
    )


@app.get("/state")
@app.get("/openenv/state")
def openenv_state():
    if _env is None:
        raise HTTPException(status_code=400, detail="Call reset before state")
    return JSONResponse(_env.state())


@app.get("/", response_class=HTMLResponse)
def home(task: str = Query(default="hard", pattern="^(easy|medium|hard)$")):
    buffer = StringIO()
    with redirect_stdout(buffer):
        run_episode(task=task, benchmark_name="email-triage")

    output = buffer.getvalue()
    html = (
        "<html><body>"
        "<h1>Email Triage OpenEnv</h1>"
        "<p>Choose a task:</p>"
        "<ul>"
        "<li><a href='/?task=easy'>easy</a></li>"
        "<li><a href='/?task=medium'>medium</a></li>"
        "<li><a href='/?task=hard'>hard</a></li>"
        "</ul>"
        "<pre style='white-space: pre-wrap;'>"
        f"{output}"
        "</pre>"
        "</body></html>"
    )
    return HTMLResponse(html)


@app.get("/raw", response_class=PlainTextResponse)
def raw(task: str = Query(default="hard", pattern="^(easy|medium|hard)$")):
    buffer = StringIO()
    with redirect_stdout(buffer):
        run_episode(task=task, benchmark_name="email-triage")
    return PlainTextResponse(buffer.getvalue())
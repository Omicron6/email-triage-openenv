from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, PlainTextResponse

from inference import run_episode


app = FastAPI(title="Email Triage OpenEnv")


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
#!/usr/bin/env python3
"""
Minimal Web Server (NO AI)
-------------------------
Serve only standalone viewers (HV check / DQM freeform) and their APIs.

This intentionally does NOT load AgentRunner / BrainAgent / WebSocket bridge.
So it can be used in environments where AI can't run.
"""

import asyncio
import os
import threading
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from tools.hv_control_tool import HVControlTool


app = FastAPI(title="autoTB Minimal Viewers (no AI)")

STATIC_DIR = Path(__file__).parent / "static"
PROJECT_ROOT = Path(__file__).parent.parent

# DQM paths
DQM_DIR = PROJECT_ROOT / "DQM"
DQM_OUTPUT_DIR = DQM_DIR / "output"
DQM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# static mounts needed by viewers
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/dqm-output", StaticFiles(directory=str(DQM_OUTPUT_DIR)), name="dqm_output")
app.mount("/jsroot", StaticFiles(directory=str(DQM_DIR)), name="jsroot")


# ──────────────────────────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    # Provide a simple landing page so users don't land on a 404.
    return JSONResponse({
        "ok": True,
        "message": "autoTB minimal viewers (no AI). Use /hv/check or /dqm/freeform",
        "hv_check_url": "/hv/check",
        "dqm_freeform_url": "/dqm/freeform",
    })

@app.get("/hv/check")
async def hv_check_page():
    return FileResponse(str(STATIC_DIR / "hv_check.html"))


@app.get("/dqm/freeform")
async def dqm_freeform():
    return FileResponse(str(STATIC_DIR / "dqm_freeform.html"))


# ──────────────────────────────────────────────────────────────────────────────
# HV APIs
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/hv/status-all")
async def api_hv_status_all(expert: bool = False):
    try:
        tool = HVControlTool()
        if not expert:
            result = tool.execute({"command": "status", "channels": "all"})
            return {"ok": True, "output": result, "expert": False}

        if not tool._ensure_connection():
            return JSONResponse({"ok": False, "error": "HV SSH connection failed"}, status_code=500)

        cmd = "./HVWrappdemo --ch all --Status --VMon --IMon --V0Set --I0Set --RUp --RDWn --SVMax"
        stdout, stderr = tool._run_remote_command(cmd)
        if not stdout or not stdout.strip():
            return JSONResponse(
                {"ok": False, "error": (stderr.strip() if stderr else "No output"), "command": cmd},
                status_code=500,
            )

        lines = [
            "📊 HV Status Query (Expert)",
            "📋 Request: Channels all",
            f"💻 Command: {cmd}",
            "",
            "📄 Output:",
            *stdout.strip().split('\n'),
        ]
        if stderr and stderr.strip():
            lines.extend(["", "⚠️ Stderr:", *stderr.strip().split('\n')])
        return {"ok": True, "output": "\n".join(lines), "expert": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/hv/expert-metrics")
async def api_hv_expert_metrics():
    """Ramp up/down and SVMax only (exact CAEN wrapper param names)."""
    try:
        tool = HVControlTool()
        if not tool._ensure_connection():
            return JSONResponse({"ok": False, "error": "HV SSH connection failed"}, status_code=500)

        rup_cmd = "./HVWrappdemo --ch all --RUp"
        rdown_cmd = "./HVWrappdemo --ch all --RDWn"
        vmax_cmd = "./HVWrappdemo --ch all --SVMax"

        def _run(cmd: str):
            stdout, stderr = tool._run_remote_command(cmd)
            return {
                "ok": bool(stdout and stdout.strip()),
                "command": cmd,
                "output": stdout.strip() if stdout else "",
                "stderr": stderr.strip() if stderr else "",
            }

        return {
            "ok": True,
            "expert_outputs": {
                "rup": _run(rup_cmd),
                "rdown": _run(rdown_cmd),
                "vmax": _run(vmax_cmd),
            },
        }
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ──────────────────────────────────────────────────────────────────────────────
# DQM APIs (for freeform viewer)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/dqm/runs")
async def api_dqm_runs():
    """List all runs available in the DQM output directory, grouped by run number."""
    import re as _re
    pattern = _re.compile(r'^Run(\d+)_(.+?)_(.+?)_((?:AuxCut_)?)(.+)\.json$')
    runs: dict[int, list] = {}
    for p in sorted(DQM_OUTPUT_DIR.glob("Run*_*.json")):
        m = pattern.match(p.name)
        if not m:
            continue
        run_num = int(m.group(1))
        type_ = m.group(2)
        method = m.group(3)
        auxcut = bool(m.group(4))
        canvas = m.group(5)
        runs.setdefault(run_num, []).append({
            "filename": p.name,
            "canvas": canvas,
            "type": type_,
            "method": method,
            "auxcut": auxcut,
            "mtime": int(p.stat().st_mtime * 1000),
        })
    result = []
    for run_num in sorted(runs.keys(), reverse=True):
        canvases = runs[run_num]
        methods = sorted(set(c["method"] for c in canvases))
        has_auxcut = any(c["auxcut"] for c in canvases)
        result.append({
            "run_number": run_num,
            "methods": methods,
            "auxcut": has_auxcut,
            "count": len(canvases),
            "canvases": canvases,
        })
    return result


class MonitRequest(BaseModel):
    run_number: int
    type: str = "full"
    method: str = "IntADC"
    modules: List[str] = []
    max_event: Optional[int] = None
    flags: List[str] = []


# ── Freeform LIVE process tracker (same behavior as main server) ──────────────
import subprocess as _subprocess
from collections import deque as _deque
_freeform_live_proc: Optional[_subprocess.Popen] = None
_freeform_live_run: Optional[int] = None
_freeform_live_lock = threading.Lock()
_freeform_live_log: _deque = _deque(maxlen=500)
_freeform_live_log_lock = threading.Lock()

import re as _re
_ANSI_ESC = _re.compile(r'\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def _strip_ansi(text: str) -> str:
    return _ANSI_ESC.sub('', text)


def _read_live_stdout(proc: "_subprocess.Popen") -> None:
    try:
        for raw in proc.stdout:
            line = _strip_ansi(raw.rstrip("\n")).lstrip("\r")
            if line:
                with _freeform_live_log_lock:
                    _freeform_live_log.append(line)
    except Exception:
        pass


@app.post("/api/dqm/run-monit")
async def api_run_monit(req: MonitRequest):
    """Execute DQM monit with custom parameters and return generated canvases."""
    global _freeform_live_proc, _freeform_live_run

    monit_bin = str(DQM_DIR / "monit")
    config_path = str(PROJECT_ROOT / "config_general.yml")

    if not Path(monit_bin).exists():
        return JSONResponse({"error": f"monit not found: {monit_bin}"}, status_code=500)

    cmd = [
        monit_bin,
        "--RunNumber", str(req.run_number),
        "--Config", config_path,
        "--type", req.type,
        "--method", req.method,
    ]
    if req.modules:
        cmd.extend(["--module"] + req.modules)
    if req.max_event and req.max_event > 0:
        cmd.extend(["--MaxEvent", str(req.max_event)])
    for flag in req.flags:
        if flag in ("LIVE", "AUXcut"):
            cmd.append(f"--{flag}")

    generated_cmd = " ".join(cmd)

    from tools.dqm_live_worker import _build_monit_env
    monit_env = _build_monit_env()

    if "LIVE" in req.flags:
        with _freeform_live_lock:
            if _freeform_live_proc is not None and _freeform_live_proc.poll() is None:
                _sentinel = DQM_OUTPUT_DIR / f"Run{_freeform_live_run}_END"
                try:
                    _sentinel.touch()
                except OSError:
                    pass
                try:
                    _freeform_live_proc.wait(timeout=10)
                except _subprocess.TimeoutExpired:
                    _freeform_live_proc.kill()
                    _freeform_live_proc.wait()
                try:
                    _sentinel.unlink(missing_ok=True)
                except OSError:
                    pass

            sentinel = DQM_OUTPUT_DIR / f"Run{req.run_number}_END"
            sentinel.unlink(missing_ok=True)
            with _freeform_live_log_lock:
                _freeform_live_log.clear()

            try:
                proc = _subprocess.Popen(
                    cmd,
                    cwd=str(DQM_DIR),
                    stdout=_subprocess.PIPE,
                    stderr=_subprocess.STDOUT,
                    env=monit_env,
                    preexec_fn=os.setpgrp,
                    text=True,
                    bufsize=1,
                )
            except FileNotFoundError as e:
                return JSONResponse({"error": str(e), "command": generated_cmd}, status_code=500)

            _freeform_live_proc = proc
            _freeform_live_run = req.run_number

            threading.Thread(
                target=_read_live_stdout, args=(proc,),
                daemon=True, name="FreeformLiveLog",
            ).start()

        return {
            "command": generated_cmd,
            "live": True,
            "run_number": req.run_number,
            "pid": proc.pid,
            "canvases": [],
        }

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(DQM_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=monit_env,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
        output = stdout.decode(errors="replace") if stdout else ""
    except asyncio.TimeoutError:
        return JSONResponse({"error": "monit timed out (5 min)", "command": generated_cmd}, status_code=504)
    except Exception as e:
        return JSONResponse({"error": str(e), "command": generated_cmd}, status_code=500)

    prefix = f"Run{req.run_number}_{req.type}_{req.method}"
    if "AUXcut" in req.flags:
        prefix += "_AuxCut"
    files = sorted(DQM_OUTPUT_DIR.glob(f"{prefix}_*.json"))
    canvases = []
    pfx = f"{prefix}_"
    for p in files:
        name = p.name
        if not name.endswith(".json"):
            continue
        canvas = name[len(pfx):-len(".json")]
        canvases.append({
            "filename": name,
            "canvas": canvas,
            "type": req.type,
            "method": req.method,
            "auxcut": "AUXcut" in req.flags,
        })

    return {
        "command": generated_cmd,
        "exit_code": proc.returncode,
        "output": output[-500:] if len(output) > 500 else output,
        "canvases": canvases,
        "live": False,
    }


@app.post("/api/dqm/kill-live")
async def api_kill_live():
    global _freeform_live_proc, _freeform_live_run

    with _freeform_live_lock:
        proc = _freeform_live_proc
        run_number = _freeform_live_run

        if proc is None or proc.poll() is not None:
            _freeform_live_proc = None
            _freeform_live_run = None
            return {"ok": True, "msg": "no live process running"}

        sentinel = DQM_OUTPUT_DIR / f"Run{run_number}_END"
        try:
            sentinel.touch()
        except OSError:
            pass

        def _wait_and_cleanup():
            try:
                proc.wait(timeout=15)
            except _subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except _subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            try:
                sentinel.unlink(missing_ok=True)
            except OSError:
                pass

        await asyncio.get_event_loop().run_in_executor(None, _wait_and_cleanup)

        _freeform_live_proc = None
        _freeform_live_run = None

    return {"ok": True, "run_number": run_number}


@app.get("/api/dqm/live-status")
async def api_live_status():
    with _freeform_live_lock:
        proc = _freeform_live_proc
        run_number = _freeform_live_run
        alive = proc is not None and proc.poll() is None
    return {"alive": alive, "run_number": run_number if alive else None}


@app.get("/api/dqm/live-log")
async def api_live_log(since: int = 0):
    with _freeform_live_log_lock:
        lines = list(_freeform_live_log)
    return {"lines": lines[since:], "total": len(lines)}


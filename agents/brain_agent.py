#!/usr/bin/env python3
"""
BrainAgent
----------
Background agent that handles ad-hoc user requests while a scenario agent
is running.  Takes natural language, infers the right tool + params via a
fine-tuned Qwen2.5-1.5B, and executes it.

Designed to stay loaded in memory at all times (server start → server stop).
"""

import json
import threading
import queue
import traceback
from typing import Dict, Any, Optional

from agents.base_agent import BaseAgent
from agents.io_handler import WebSocketIO


# ── Tool registry (lazy imports to avoid circular deps) ──────────────────────

def _get_tool(name: str):
    """Return a tool instance by name.  Imported lazily."""
    if name == "daq_run":
        from tools.daq_tool import DAQRunTool
        return DAQRunTool()
    if name == "dqm_plot":
        from tools.dqm_tool import DQMPlotTool
        return DQMPlotTool()
    if name == "run_log":
        from tools.run_log_tool import RunLogTool
        return RunLogTool()
    if name == "hv_read":
        from tools.hv_control_tool import HVControlTool
        return HVControlTool()
    return None


# Map tool names to shared lock keys.  None means no lock needed.
TOOL_LOCK_MAP = {
    "daq_run": "daq",
    "dqm_plot": None,
    "run_log": None,
    "hv_read": "hv",
    "hv_write": "hv",
}

# Tools that require user confirmation before execution.
TOOLS_NEED_CONFIRM = {"hv_write"}


# ── Regex fallback for critical HV-write commands ────────────────────────────
#  Runs BEFORE the LLM output is used so that even an un-retrained model
#  cannot silently miss a dangerous HV request.

_CHANNEL_RE = r'T[1-9][CScs]'                                     # e.g. T1C, T3S
_VOLTAGE_RE = r'(\d+(?:\.\d+)?)\s*[Vv]?'                          # 1500, 1500V, 100
# "전압" / "전압을" / "전압으로" can appear between channel name and number
_VOLT_KW    = r'(?:전압\s*(?:을|이|으로)?\s*)?'
_ACTION_RE  = r'(?:(?:으로|로)\s*)?(?:수정|변경|설정|바꿔|올려|내려|인가|해줘|맞춰|세팅)?'


def _pattern_match_hv_write(user_input: str) -> Optional[dict]:
    """Return a hv_write decision if the input clearly matches, else None."""
    s = user_input.strip()

    # 1) Single/multi channel with voltage:
    #    "T1S 20V로 수정", "T1C, T2C 1500V로 바꿔", "T9S 전압 100으로"
    m = re.search(
        rf'({_CHANNEL_RE}(?:\s*[,/]\s*{_CHANNEL_RE})*)\s+{_VOLT_KW}{_VOLTAGE_RE}{_ACTION_RE}',
        s,
    )
    if m:
        chs_raw = m.group(1)
        voltage = float(m.group(2))
        channels = [c.strip().upper() for c in re.split(r'[,/]', chs_raw)]
        return {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": channels, "voltage": voltage},
            "reason": f"{', '.join(channels)} 채널 전압을 {voltage}V로 변경",
        }

    # 2) All channels voltage change: "HV 1500V로 수정", "전체 HV 1500V"
    #    Require an explicit all-channel keyword AND a verb — channel-specific requests
    #    must NOT match here even if they fall through from rule 1.
    m = re.search(
        rf'(?:HV|전체|모든|전\s*채널)\s*(?:전압\s*(?:을)?\s*)?{_VOLTAGE_RE}',
        s, re.IGNORECASE,
    )
    if m and re.search(r'(?:수정|변경|설정|바꿔|올려|내려|맞춰|세팅)', s):
        voltage = float(m.group(1))
        return {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": "all", "voltage": voltage},
            "reason": f"모든 채널 전압을 {voltage}V로 변경",
        }

    # 3) ON / OFF
    if re.search(r'HV.*(?:켜|on\b|turn\s*on)', s, re.IGNORECASE):
        return {
            "tool": "hv_write",
            "params": {"command": "on", "channels": "all"},
            "reason": "모든 HV 채널 켜기",
        }
    if re.search(r'HV.*(?:꺼|off\b|turn\s*off)', s, re.IGNORECASE):
        return {
            "tool": "hv_write",
            "params": {"command": "off", "channels": "all"},
            "reason": "모든 HV 채널 끄기",
        }

    return None


SYSTEM_PROMPT = """You are the Brain Agent for a test beam experiment (KEK/CERN).
Your job is to interpret the operator's ad-hoc request and call the right tool.

Available tools:
- daq_run: Run DAQ data collection. params: {"events": int}
- dqm_plot: Generate DQM plots for a run and display in the DQM panel.
  params: {"run_number": int, "method": "IntADC"|"PeakADC", "type": "full"|"heatmap"|"single", "modules": [list]}
  - type defaults to "full" (all towers + heatmap). No modules needed for full.
  - "heatmap": modules must be ["MCPPMT"]. method: IntADC or PeakADC.
  - "single": modules is a list of channel names, e.g. ["T1-C"], ["T1-S","T1-C"], or ["T1"] (T1 auto-expands).
  - method defaults to "IntADC". Use "PeakADC" only when explicitly requested.
- run_log: Google Sheets run log.
  Read:   params: {"command": "read", "run_num": int}
  Update: params: {"command": "update", "run_num": int, "<column>": "<value>"}
  Updatable columns: program, notes, config, beam_energy, beam_type, trigger_setup, hv_drc, hv_aux
- hv_read: Read current HV status. params: {"command": "status"} (optional: "channels": "all" | list)
- hv_write: Change HV voltage or turn channels on/off. User confirmation will be asked before execution.
  Voltage: {"command": "voltage", "channels": ["T1C", ...] | "all", "voltage": <V as float>}
  On/off:  {"command": "on"|"off", "channels": ["T1C", ...] | "all"}
  Channel names: T1C, T1S, T2C, T2S, ..., T9C, T9S

Current experiment state is provided so you can resolve relative references
like "방금", "이번 런", "지금" to concrete run numbers or energies.

Respond with a single JSON object:
{"tool": "<tool_name>", "params": {<params>}, "reason": "<short explanation>"}

If the request is unclear or you cannot determine a tool, respond:
{"tool": "none", "message": "<ask the user for clarification>"}

RULES:
1. Output ONLY valid JSON. No markdown, no explanation outside JSON.
2. Always resolve relative references using the provided state.
3. For run_log updates, extract column and value from the user's message.
4. run_log supports both READ and WRITE:
   - VIEW/CHECK a log (확인, 보여줘, 읽어줘) WITHOUT a value → {"command": "read", "run_num": ...}
   - WRITE with column+value (e.g. "프로그램에 EM 추가") → {"command": "update", "run_num": ..., "<column>": "<value>"}
5. hv_read CAN read. "HV 확인" → use hv_read.
6. DAQ requires an event count. If the user says "DAQ 돌려줘" without a number, ask how many events.
7. Channel names like T1C, T1S, T2C, ..., T9S are HV channels — NOT log columns.
   "T9S 전압 100으로", "T1C 1500V로 수정" → hv_write with channels: ["T9S"] or ["T1C"].
   ONLY use channels: "all" when the input explicitly says 전체/모든/전 채널/all channels.
   A SINGLE channel name + voltage ALWAYS means channels: [that single channel].
8. For hv_write, the system asks the user to confirm before execution — you don't need to handle confirmation in your JSON.
9. "플롯", "그려줘", "그래프", "확인해줘 (run)" → dqm_plot. Default type: full, default method: IntADC.
10. Specific tower/channel (T1, T1-C, T1-S, T5 etc.) → type: single, modules: [name].
11. "heatmap" or "MCPPMT" mentioned → type: heatmap, modules: ["MCPPMT"]. Always MCPPMT (SiPM not used).
12. No type/channel hint → type: full.
"""


class BrainAgent(BaseAgent):
    """
    Lightweight agent for ad-hoc tool dispatch.
    - Stays loaded in memory (no context-manager cycling)
    - Single-turn: one request  ->  one tool call  ->  result
    - Reads scenario agent state (read-only) for context
    """

    def __init__(self, shared_state: Optional[Dict] = None,
                 shared_locks: Optional[Dict[str, threading.Lock]] = None,
                 io_handler=None, use_base_model: bool = False,
                 confirm_queue: Optional[queue.Queue] = None,
                 clarify_queue: Optional[queue.Queue] = None):
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from config import AGENT_MODELS

        model_cfg = AGENT_MODELS["brain"]
        model_path = model_cfg["base_model"] if use_base_model else model_cfg["fine_tuned_path"]

        super().__init__(model_path=model_path, agent_name="BrainAgent", io_handler=io_handler)

        self.shared_state = shared_state or {}
        self.shared_locks = shared_locks or {}
        self.confirm_queue = confirm_queue or queue.Queue()
        self.clarify_queue = clarify_queue or queue.Queue()

    # ── BaseAgent abstract methods ───────────────────────────────────────────

    def _get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def _build_state_context(self) -> str:
        """Summarise the scenario agent's state for context."""
        s = self.shared_state
        if not s:
            return "(No scenario agent running)"

        lines = []
        if s.get("agent_type"):
            lines.append(f"Running agent: {s['agent_type']}")
        if s.get("current_run"):
            lines.append(f"Current run number: {s['current_run']}")
        if s.get("last_run"):
            lines.append(f"Last completed run: {s['last_run']}")
        if s.get("current_tower"):
            lines.append(f"Current tower: {s['current_tower']}")
        if s.get("current_energy"):
            lines.append(f"Current energy: {s['current_energy']} GeV")
        if s.get("phase"):
            lines.append(f"Phase: {s['phase']}")
        return "\n".join(lines) if lines else "(No scenario agent running)"

    def run(self):
        """Not used — BrainAgent uses handle_request() for single-turn dispatch."""
        pass

    # ── Public API ───────────────────────────────────────────────────────────

    def handle_request(self, user_input: str, io: WebSocketIO) -> None:
        """
        Process one ad-hoc request end-to-end:
        1. Build context  (state + user input)
        2. LLM inference  (tool + params)
        3. Lock check
        4. Execute tool
        5. Send result to UI
        """
        # 1. Build context
        context = self.build_full_context(current_input=user_input)

        # 2. LLM inference
        io.send_status("BrainAgent 처리 중...")
        decision = self.decide(context)

        if "error" in decision:
            io.send_ai_message(f"요청을 이해하지 못했습니다: {decision.get('raw_output', '')[:200]}")
            io.send_status("대기 중")
            return

        tool_name = decision.get("tool", "none")
        reason = decision.get("reason", "")

        # "none" → clarification message
        if tool_name == "none":
            msg = decision.get("message", reason or "무엇을 도와드릴까요?")
            io.send_ai_message(msg)
            io.send_status("대기 중")
            return

        params = decision.get("params", {})

        # 2.5. Parameter validation — if something is missing, ask via popup
        #      and retry up to 2 times.
        for _attempt in range(2):
            missing = self._validate_params(tool_name, params)
            if not missing:
                break
            field_key, question = missing
            answer = self._ask_clarify(question, io)
            if not answer:
                io.send_ai_message("확인 대기 시간이 초과되어 요청을 취소했습니다.")
                io.send_status("대기 중")
                return
            parsed = self._parse_clarify_answer(field_key, answer)
            if parsed is None:
                io.send_ai_message("답변을 이해하지 못했습니다. 요청을 취소합니다.")
                io.send_status("대기 중")
                return
            params[field_key] = parsed
        else:
            # still missing after retries
            io.send_ai_message("필수 정보가 누락되어 요청을 취소합니다.")
            io.send_status("대기 중")
            return

        # 2.6. Confirmation required for dangerous tools
        if tool_name in TOOLS_NEED_CONFIRM:
            preview = self._format_confirm_preview(tool_name, params)
            io.send_ai_message(f"{reason}")
            # Drain any stale confirmation responses
            while not self.confirm_queue.empty():
                try: self.confirm_queue.get_nowait()
                except queue.Empty: break
            # Send confirmation request to popup UI
            io.output_queue.put({
                "type": "adhoc_confirm",
                "preview": preview,
                "tool": tool_name,
            })
            io.send_status("사용자 확인 대기 중...")
            # Block until user clicks 확인/취소
            try:
                confirmed = self.confirm_queue.get(timeout=120)
            except queue.Empty:
                io.send_ai_message("확인 대기 시간 초과 — 요청을 취소했습니다.")
                io.send_status("대기 중")
                return
            if not confirmed:
                io.send_ai_message("요청이 취소되었습니다.")
                io.send_status("대기 중")
                return
            # Fall through to execution
        else:
            io.send_ai_message(f"{reason}")

        # 3. Lock check
        lock_key = TOOL_LOCK_MAP.get(tool_name)
        lock = self.shared_locks.get(lock_key) if lock_key else None

        if lock and not lock.acquire(blocking=False):
            resource = {"daq": "DAQ", "hv": "HV"}.get(lock_key, lock_key)
            io.send_ai_message(f"{resource}가 현재 사용 중이라 지금은 실행할 수 없습니다.")
            io.send_status("대기 중")
            return

        # 4. Execute tool
        try:
            self._execute_tool(tool_name, params, io)
        finally:
            if lock:
                lock.release()

        io.send_status("대기 중")

    @staticmethod
    def _format_confirm_preview(tool_name: str, params: dict) -> str:
        """Human-readable summary of the action, shown in the confirmation popup."""
        if tool_name == "hv_write":
            cmd = params.get("command", "?")
            ch = params.get("channels", "?")
            ch_str = ch if isinstance(ch, str) else ", ".join(ch)
            if cmd == "voltage":
                v = params.get("voltage", "?")
                return f"HV 전압 변경\n  채널: {ch_str}\n  전압: {v} V"
            if cmd == "current":
                c = params.get("current", "?")
                return f"HV 전류 변경\n  채널: {ch_str}\n  전류: {c} μA"
            if cmd == "on":
                return f"HV 켜기\n  채널: {ch_str}"
            if cmd == "off":
                return f"HV 끄기\n  채널: {ch_str}"
        return f"{tool_name}\n  params: {params}"

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_params(tool_name: str, params: dict):
        """
        Return (field_key, question) for the first missing required param,
        or None if all required params are present.
        field_key is used to parse the clarification answer.
        """
        if tool_name == "daq_run":
            events = params.get("events")
            if not events or (isinstance(events, (int, float)) and events <= 0):
                return ("events", "몇 개의 이벤트를 수집할까요?")
        if tool_name == "dqm_plot":
            if not params.get("run_number"):
                return ("run_number", "어떤 런 번호의 DQM 플롯을 그릴까요?")
        if tool_name == "run_log":
            if not params.get("run_num"):
                return ("run_num", "어떤 런 번호의 로그를 처리할까요?")
        return None

    @staticmethod
    def _parse_clarify_answer(field_key: str, answer: str):
        """Extract the required value from user's clarification answer."""
        import re
        s = (answer or "").strip()
        if field_key in ("events",):
            m = re.search(r'(\d+)', s)
            return int(m.group(1)) if m else None
        if field_key in ("run_number", "run_num"):
            m = re.search(r'(\d{3,7})', s)
            return int(m.group(1)) if m else None
        return s

    def _ask_clarify(self, question: str, io, timeout: int = 120) -> Optional[str]:
        """Send a clarification request to the popup and wait for the answer."""
        # Drain stale clarify replies
        while not self.clarify_queue.empty():
            try: self.clarify_queue.get_nowait()
            except queue.Empty: break
        io.output_queue.put({"type": "adhoc_clarify", "question": question})
        io.send_status("추가 정보 입력 대기 중...")
        try:
            return self.clarify_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── Internal ─────────────────────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, params: dict, io: WebSocketIO):
        """Run the actual tool and send output to the UI."""
        try:
            if tool_name == "daq_run":
                from tools.daq_tool import DAQRunTool
                io.send_status("DAQ 실행 중...")
                DAQRunTool().execute(params, line_callback=io.send_tool_output)

            elif tool_name == "dqm_plot":
                from tools.dqm_tool import DQMPlotTool
                from agents.dqm_live_worker import OUTPUT_DIR
                run_number = int(params.get("run_number", 0))
                method = params.get("method", "IntADC")
                type_ = params.get("type", "full")
                modules = params.get("modules", [])
                io.send_status("DQM 플롯 생성 중...")

                result = DQMPlotTool().execute(params)
                io.send_tool_output(result)

                # Determine base_prefix for JSON scanning (mirrors TBplotengine.cc naming)
                if type_ == "full":
                    base_prefix = f"Run{run_number}_full_{method}"
                elif type_ in ("heatmap", "module"):
                    mod = modules[0] if modules else "MCPPMT"
                    base_prefix = f"Run{run_number}_{type_}_{method}_{mod}"
                else:
                    # single: fModule="" in C++ → trailing underscore in basePrefix
                    base_prefix = f"Run{run_number}_single_{method}_"

                pfx = f"{base_prefix}_"
                canvases = [
                    p.name[len(pfx):-len(".json")]
                    for p in sorted(OUTPUT_DIR.glob(f"{base_prefix}_*.json"))
                    if p.name.endswith(".json")
                ]
                if canvases:
                    io.output_queue.put({
                        "type": "dqm_canvases",
                        "base_prefix": base_prefix,
                        "canvases": canvases,
                        "run_number": run_number,
                    })
                else:
                    io.send_ai_message(f"Run {run_number} DQM JSON 파일을 찾을 수 없습니다.")

            elif tool_name in ("run_log", "run_log_read"):
                from tools.run_log_tool import RunLogTool
                result = RunLogTool().execute(params)
                io.send_tool_output(result)

            elif tool_name == "hv_read":
                from tools.hv_control_tool import HVControlTool
                result = HVControlTool().execute(params)
                io.send_tool_output(result)

            elif tool_name == "hv_write":
                from tools.hv_control_tool import HVControlTool
                io.send_status("HV 변경 중...")
                result = HVControlTool().execute(params)
                io.send_tool_output(result)

            else:
                io.send_ai_message(f"알 수 없는 도구: {tool_name}")

        except Exception as e:
            io.send_ai_message(f"도구 실행 중 오류: {e}")


# ── Background worker loop ───────────────────────────────────────────────────

class _BrainOutputQueue:
    """Wrapper that tags every message with source='brain' before forwarding."""

    def __init__(self, real_queue: queue.Queue):
        self._q = real_queue

    def put(self, item):
        if isinstance(item, dict):
            item = {**item, "source": "brain"}
        self._q.put(item)

    def get(self, *args, **kwargs):
        return self._q.get(*args, **kwargs)

    def get_nowait(self):
        return self._q.get_nowait()

    def empty(self):
        return self._q.empty()


def run_brain_thread(
    brain_agent: BrainAgent,
    adhoc_queue: queue.Queue,
    output_queue: queue.Queue,
    stop_event: threading.Event,
    confirm_queue: Optional[queue.Queue] = None,
    clarify_queue: Optional[queue.Queue] = None,
):
    """
    Long-running thread that polls adhoc_queue and dispatches requests
    through BrainAgent.  Shares output_queue with the scenario agent
    so results appear in the same UI.
    """
    if confirm_queue is not None:
        brain_agent.confirm_queue = confirm_queue
    if clarify_queue is not None:
        brain_agent.clarify_queue = clarify_queue
    tagged_queue = _BrainOutputQueue(output_queue)
    io = WebSocketIO(
        input_queue=queue.Queue(),   # BrainAgent doesn't need input back
        output_queue=tagged_queue,
        stop_event=stop_event,
    )

    while not stop_event.is_set():
        try:
            user_input = adhoc_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if user_input in ("종료", "exit"):
            break

        try:
            brain_agent.handle_request(user_input, io)
        except Exception:
            tb = traceback.format_exc()
            output_queue.put({"type": "error", "content": f"BrainAgent error:\n{tb}"})

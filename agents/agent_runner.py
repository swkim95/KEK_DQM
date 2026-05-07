#!/usr/bin/env python3
"""
AgentRunner
-----------
Runs a specialized agent in a background thread.
Bridges between the async FastAPI WebSocket and the blocking agent .run() call.

input_queue  : WebSocket → agent  (user sends "완료" or free text)
output_queue : agent → WebSocket  (AI messages, tool output, plots, status)
adhoc_queue  : WebSocket → BrainAgent  (ad-hoc requests during scenario run)

BrainAgent stays loaded at all times and handles ad-hoc requests in a
separate thread, sharing output_queue with the scenario agent.
"""

import json
import queue
import re
import threading
import traceback
from typing import Optional

from agents.io_handler import WebSocketIO


def _parse_energy_config(text: str) -> dict:
    """
    Parse user energy input into {energy_GeV: n_events} dict.
    Handles common formats:
      "20:1000 40:1000"
      "1GeV 100개 2GeV 100개"
      "20, 40GeV 각각 1000개"
    Returns empty dict if parsing fails.
    """
    # Format 1: "20:1000 40:1000"
    colon_pairs = re.findall(r'(\d+(?:\.\d+)?)\s*:\s*(\d+)', text)
    if colon_pairs:
        return {int(float(e)): int(n) for e, n in colon_pairs}

    # Format 2: "1GeV 100개 2GeV 100개"  (energy-events pairs)
    gev_pairs = re.findall(
        r'(\d+(?:\.\d+)?)\s*[Gg][Ee][Vv]?\s+(\d+)\s*(?:개|events?)?', text
    )
    if gev_pairs:
        return {int(float(e)): int(n) for e, n in gev_pairs}

    # Format 3: "20, 40GeV 각각 1000개"  (multiple energies, shared event count)
    energies = re.findall(r'(\d+(?:\.\d+)?)\s*[Gg][Ee][Vv]?', text)
    events_m = re.search(r'(\d+)\s*(?:개|events?)', text)
    if energies and events_m:
        evts = int(events_m.group(1))
        return {int(float(e)): evts for e in energies}

    return {}


# ── Shared resource locks (hardware collision prevention) ────────────────────

shared_locks = {
    "daq": threading.Lock(),
    "hv": threading.Lock(),
}

# ── Shared state (scenario agent writes, BrainAgent reads) ──────────────────
# This dict is replaced wholesale when a scenario agent starts (pointing to
# agent.state), and cleared when it stops.  BrainAgent reads it (no lock
# needed for dict reads in CPython due to GIL).

shared_state: dict = {}


def update_shared_state(updates: dict):
    """Merge updates into shared_state."""
    shared_state.update(updates)


def set_shared_state_ref(agent_state: dict):
    """Replace shared_state contents with a live reference to agent.state fields.
    Called once after agent creation so BrainAgent always sees current values.
    Preserves runner-injected keys (agent_type, _output_queue) that downstream
    tools rely on regardless of which sub-agent is currently active."""
    global shared_state
    preserved = {k: shared_state[k] for k in ("agent_type", "_output_queue")
                 if k in shared_state}
    shared_state.clear()
    shared_state.update(agent_state)
    shared_state.update(preserved)
    # Keep a back-reference so periodic sync can refresh
    shared_state["_agent_state_ref"] = agent_state


def sync_shared_state():
    """Copy key fields from the agent's state into shared_state."""
    ref = shared_state.get("_agent_state_ref")
    if ref is None:
        return
    for key in ("current_energy", "current_tower", "last_run_number",
                "phase", "current_energy_idx", "scan_order"):
        if key in ref:
            shared_state[key] = ref[key]
    # Alias for BrainAgent convenience
    if "last_run_number" in ref:
        shared_state["current_run"] = ref["last_run_number"]


def clear_shared_state():
    """Clear shared_state when scenario agent stops."""
    shared_state.clear()


def run_agent_thread(
    agent_name: str,
    params: dict,
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    stop_event: threading.Event,
    waiting_flag: threading.Event = None,
):
    io = WebSocketIO(input_queue, output_queue, stop_event, waiting_flag=waiting_flag)

    try:
        io.send_status(f"{agent_name} 에이전트 시작 중...")
        # _output_queue is consumed by DAQRunTool to push DQM live events
        # back to the browser without taking io_handler as a dependency.
        update_shared_state({"agent_type": agent_name, "_output_queue": output_queue})

        # ── EM Scan ──────────────────────────────────────────────────────────
        if agent_name == "em_scan":
            from agents.energy_scan_agent import EnergyScanAgent

            # Ask for energy config, parse with Python (don't rely on base LLM)
            energy_config = {}
            while not energy_config:
                io.send_ai_message(
                    "에너지 설정을 입력해주세요.\n"
                    "예) 20:1000 40:1000   또는   1GeV 100개 2GeV 100개"
                )
                energy_input = io.get_input()
                if energy_input in ("종료", "exit"):
                    output_queue.put({"type": "agent_done"})
                    return
                energy_config = _parse_energy_config(energy_input)
                if not energy_config:
                    io.send_ai_message(
                        "⚠️ 형식을 인식하지 못했습니다. 다시 입력해주세요.\n"
                        "예) 20:1000 40:1000"
                    )

            agent = EnergyScanAgent(
                energy_config=energy_config,   # parsed dict → phase starts "idle"
                use_base_model=params.get("use_base_model", False),
                io_handler=io,
            )

            # Inject synthetic STEP-0 history so the model sees energy config was
            # already parsed by Python and jumps straight to STEP 1 (T5 movement).
            # Without this, a retrained model may repeat STEP 0 or ask for energy again.
            energy_config_snapshot = {
                str(k): {
                    "target_events": v, "collected_events": 0,
                    "runs": [], "completed": False, "completed_at": None,
                }
                for k, v in energy_config.items()
            }
            agent.add_to_history("user", energy_input)
            agent.add_to_history("assistant", json.dumps({
                "tool": "none",
                "update_state": {
                    "energy_config": energy_config_snapshot,
                    "scan_order": sorted(energy_config.keys()),
                    "phase": "idle",
                },
            }, ensure_ascii=False))

        # ── Calibration Scan ─────────────────────────────────────────────────
        elif agent_name == "calib_scan":
            from agents.calib_scan_agent import CalibScanAgent

            agent = CalibScanAgent(
                beam_energy=None,
                target_events=None,
                use_base_model=params.get("use_base_model", False),
                io_handler=io,
            )

        # ── HV Equalization ──────────────────────────────────────────────────
        elif agent_name in ("hv_equalization", "hv_equalization_sim"):
            _HV_TOWER_ORDER = ["T1", "T2", "T3", "T6", "T5", "T4", "T7", "T8", "T9"]
            _is_sim = (agent_name == "hv_equalization_sim")

            # Config 수집 (Python에서 한 번만)
            def _ask_float(prompt):
                while True:
                    io.send_ai_message(prompt)
                    val = io.get_input()
                    if val in ("종료", "exit"):
                        return None
                    try:
                        return float(val.strip())
                    except ValueError:
                        io.send_ai_message("⚠️ 숫자를 입력해주세요.")

            def _ask_int(prompt):
                v = _ask_float(prompt)
                return int(v) if v is not None else None

            beam_energy = _ask_float("빔 에너지 (GeV)를 입력해주세요.")
            if beam_energy is None:
                output_queue.put({"type": "agent_done"}); return
            target_events = _ask_int(f"이벤트 수를 입력해주세요.")
            if target_events is None:
                output_queue.put({"type": "agent_done"}); return
            target_adc = _ask_float("목표 peakADC 값을 입력해주세요.")
            if target_adc is None:
                output_queue.put({"type": "agent_done"}); return

            update_shared_state({
                "current_energy": beam_energy,
            })

            # HV 세션 초기화
            from tools.hv_equalization_tool import hv_equalization_start
            result = hv_equalization_start.invoke({
                "target_c": target_adc, "target_s": target_adc,
                "tower": _HV_TOWER_ORDER[0],
            }) if hasattr(hv_equalization_start, "invoke") else hv_equalization_start(
                target_c=target_adc, target_s=target_adc, tower=_HV_TOWER_ORDER[0]
            )
            io.send_tool_output(result)

            # 타워 루프
            AgentClass = None
            if _is_sim:
                from agents.hv_equalization_sim_agent import HVEqualizationSimAgent
                AgentClass = HVEqualizationSimAgent
            else:
                from agents.hv_equalization_agent import HVEqualizationAgent
                AgentClass = HVEqualizationAgent

            for i, tower in enumerate(_HV_TOWER_ORDER):
                if stop_event.is_set():
                    break
                io.send_status(f"[{i + 1}/{len(_HV_TOWER_ORDER)}] {tower} 타워 시작")
                update_shared_state({"current_tower": tower, "agent_type": agent_name})
                tower_agent = AgentClass(
                    tower=tower,
                    beam_energy=beam_energy,
                    target_events=target_events,
                    target_adc=target_adc,
                    use_base_model=params.get("use_base_model", False),
                    io_handler=io,
                )
                set_shared_state_ref(tower_agent.state)
                _hv_sync_stop = threading.Event()
                def _hv_sync(evt=_hv_sync_stop):
                    while not evt.is_set():
                        sync_shared_state()
                        evt.wait(1.0)
                _hv_sync_t = threading.Thread(target=_hv_sync, daemon=True)
                _hv_sync_t.start()
                with tower_agent:
                    tower_agent.run()
                _hv_sync_stop.set()
                io.send_status(f"✅ {tower} 완료")

            io.send_ai_message("모든 타워 HV Equalization이 완료되었습니다.")
            io.send_status("완료")
            clear_shared_state()
            output_queue.put({"type": "agent_done"})
            return

        else:
            output_queue.put({"type": "error", "content": f"Unknown agent: {agent_name}"})
            output_queue.put({"type": "agent_done"})
            return

        set_shared_state_ref(agent.state)
        # Periodic sync thread keeps shared_state fresh for BrainAgent
        _sync_stop = threading.Event()
        def _sync_loop():
            while not _sync_stop.is_set():
                sync_shared_state()
                _sync_stop.wait(1.0)
        _sync_thread = threading.Thread(target=_sync_loop, daemon=True)
        _sync_thread.start()
        with agent:
            agent.run()
        _sync_stop.set()

        io.send_status("완료")
        clear_shared_state()
        output_queue.put({"type": "agent_done"})

    except StopAgentException:
        clear_shared_state()
        output_queue.put({"type": "status", "content": "에이전트가 중지되었습니다."})
        output_queue.put({"type": "agent_done"})

    except Exception as e:
        clear_shared_state()
        tb = traceback.format_exc()
        output_queue.put({"type": "error", "content": f"Agent error: {str(e)}\n{tb}"})
        output_queue.put({"type": "agent_done"})


class StopAgentException(Exception):
    """Raised inside the agent thread to break out of the run loop."""
    pass


class AgentRunner:
    """
    Manages a single running scenario agent thread + a persistent BrainAgent
    background thread for ad-hoc requests.
    """

    def __init__(self):
        # Scenario agent
        self.thread: Optional[threading.Thread] = None
        self.input_queue: queue.Queue = queue.Queue()
        self.output_queue: queue.Queue = queue.Queue()
        self.stop_event: threading.Event = threading.Event()
        self.waiting_flag: threading.Event = threading.Event()  # set when scenario blocks on get_input()

        # BrainAgent (ad-hoc)
        self.adhoc_queue: queue.Queue = queue.Queue()
        self.confirm_queue: queue.Queue = queue.Queue()   # Yes/No replies from popup
        self.clarify_queue: queue.Queue = queue.Queue()   # follow-up text replies from popup
        self._brain_agent = None
        self._brain_thread: Optional[threading.Thread] = None
        self._brain_stop = threading.Event()

    @property
    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    @property
    def brain_ready(self) -> bool:
        return (self._brain_agent is not None
                and self._brain_agent.model is not None
                and self._brain_thread is not None
                and self._brain_thread.is_alive())

    def start_brain(self, use_base_model: bool = False):
        """Load BrainAgent and start its background thread."""
        if self.brain_ready:
            return

        from agents.brain_agent import BrainAgent, run_brain_thread

        self._brain_stop.clear()
        self._brain_agent = BrainAgent(
            shared_state=shared_state,
            shared_locks=shared_locks,
            use_base_model=use_base_model,
        )
        self._brain_agent.load()

        self._brain_thread = threading.Thread(
            target=run_brain_thread,
            args=(self._brain_agent, self.adhoc_queue, self.output_queue,
                  self._brain_stop, self.confirm_queue, self.clarify_queue),
            daemon=True,
        )
        self._brain_thread.start()

    def stop_brain(self):
        """Stop BrainAgent thread and unload model."""
        self._brain_stop.set()
        if self._brain_thread and self._brain_thread.is_alive():
            self._brain_thread.join(timeout=5)
        if self._brain_agent:
            self._brain_agent.unload()
            self._brain_agent = None

    def send_adhoc(self, text: str):
        """Send an ad-hoc request to BrainAgent."""
        self.adhoc_queue.put(text)

    def send_confirm(self, confirmed: bool):
        """Send a Yes/No reply to BrainAgent's pending confirmation."""
        self.confirm_queue.put(confirmed)

    def send_clarify(self, text: str):
        """Send a follow-up clarification answer to BrainAgent."""
        self.clarify_queue.put(text)

    def start(self, agent_name: str, params: dict):
        if self.is_running:
            raise RuntimeError("An agent is already running.")

        # Clear queues and reset stop flag
        while not self.input_queue.empty():
            self.input_queue.get_nowait()
        while not self.output_queue.empty():
            self.output_queue.get_nowait()
        while not self.adhoc_queue.empty():
            self.adhoc_queue.get_nowait()
        self.stop_event.clear()

        self.waiting_flag.clear()
        self.thread = threading.Thread(
            target=run_agent_thread,
            args=(agent_name, params, self.input_queue, self.output_queue,
                  self.stop_event, self.waiting_flag),
            daemon=True,
        )
        self.thread.start()

    def send_input(self, text: str):
        self.input_queue.put(text)

    def stop(self):
        """Create KILLME to stop any running DAQ, then signal the agent thread."""
        # 1. Kill DAQ run first
        try:
            from tools.daq_tool import KILLME_FILE
            from pathlib import Path
            Path(KILLME_FILE).touch()
        except Exception:
            pass
        # 2. Stop the agent thread
        self.stop_event.set()
        self.input_queue.put("종료")

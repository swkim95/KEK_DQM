#!/usr/bin/env python3
"""
Training data generator for BrainAgent
- Single-turn: state + user request  →  tool call JSON
- Covers all tool types × expression variety × state combinations
- Target: daq_run 150, dqm_plot 300, run_log 150+70, hv_read 150, hv_write 110, none 150
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any


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


# ── State builder (mirrors brain_agent.py) ───────────────────────────────────

def _build_state_context(state: dict) -> str:
    if not state:
        return "(No scenario agent running)"
    lines = []
    if state.get("agent_type"):
        lines.append(f"Running agent: {state['agent_type']}")
    if state.get("current_run"):
        lines.append(f"Current run number: {state['current_run']}")
    if state.get("last_run"):
        lines.append(f"Last completed run: {state['last_run']}")
    if state.get("current_tower"):
        lines.append(f"Current tower: {state['current_tower']}")
    if state.get("current_energy"):
        lines.append(f"Current energy: {state['current_energy']} GeV")
    if state.get("phase"):
        lines.append(f"Phase: {state['phase']}")
    return "\n".join(lines) if lines else "(No scenario agent running)"


def build_full_context(state: dict, user_input: str) -> str:
    parts = []
    parts.append("=== Current State ===")
    parts.append(_build_state_context(state))
    parts.append("")
    parts.append("=== Recent Conversation ===")
    parts.append("(No conversation yet)")
    parts.append("")
    parts.append("=== Current User Input ===")
    parts.append(user_input)
    parts.append("")
    parts.append("=== Your Task ===")
    parts.append("Based on the current state and conversation, decide the next action.")
    parts.append("Output JSON with tool name and parameters.")
    return "\n".join(parts)


def make_example(state: dict, user_input: str, decision: dict) -> dict:
    ctx = build_full_context(state, user_input)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ctx},
            {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)},
        ]
    }


# ── Random state generators ─────────────────────────────────────────────────

def _random_run():
    # 더 넓은 런 번호 범위 (실제 실험 런 다양성 반영)
    return random.randint(10000, 15000)

def _random_energy():
    return random.choice([1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 80, 100, 120])

def _random_tower():
    return random.choice(["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"])

def _random_events():
    return random.choice([100, 200, 500, 1000, 2000, 3000, 5000, 10000, 20000])


def _make_state(with_agent=True):
    if not with_agent:
        return {}
    run = _random_run()
    return {
        "agent_type": random.choice(["energy_scan", "calib_scan", "hv_equalization"]),
        "current_run": run,
        "last_run": run - 1,
        "current_tower": _random_tower(),
        "current_energy": _random_energy(),
        "phase": random.choice(["scanning", "daq_running", "waiting_confirm", "hv_adjusting"]),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DAQ RUN — 100 samples
# ══════════════════════════════════════════════════════════════════════════════

def gen_daq_run() -> List[dict]:
    examples = []
    templates = [
        # 기본
        "{n}개 돌려줘", "{n}개 이벤트 받아줘", "이벤트 {n}개 수집해줘",
        "DAQ {n}개 돌려", "데이터 {n}개만 받자", "{n}개만 더 받아줘",
        "{n}개 추가로 받아줘", "DAQ 한번 돌려줘 {n}개", "{n}개 빨리 돌려",
        "{n}개 이벤트 수집", "데이터 수집 {n}개", "이벤트 {n}개 돌려줘",
        "{n}개 데이터 수집해줘", "{n}개 받자",
        # 영어 혼용
        "run {n} events", "DAQ run {n}", "{n} events 돌려줘",
        "{n} events please", "take {n} events", "collect {n} events",
        "{n} events 받아줘", "daq {n}", "{n} evt",
        # 존댓말
        "{n}개 이벤트 수집 부탁드립니다", "이벤트 {n}개 받아주세요",
        "DAQ를 {n}개 돌려주세요", "{n}개 데이터 받아주실 수 있나요",
        "{n}개 이벤트 수집해주세요", "데이터 {n}개 수집 부탁해요",
        "{n}개만 받아주시겠어요",
        # 반말 / 채팅체
        "{n}개 받아", "{n}개 돌려", "데이터 {n}개", "{n}개 ㄱㄱ",
        "이벤트 {n}개 좀", "{n}개 ㄱ", "{n}개 달려", "{n} 돌려",
        # 수량 변형
        "{n}개만 받아줘", "{n}개만 돌려", "{n}개 정도 돌려줘",
        "{n}개쯤 받자", "한 {n}개 돌려볼까", "약 {n}개 받아줘",
        "{n}개만 좀 받아줘", "{n}개 정도만", "딱 {n}개만",
        # 목적/맥락
        "테스트로 {n}개 돌려줘", "확인용으로 {n}개만", "pedestal {n}개 받아줘",
        "빔 데이터 {n}개 받자", "노이즈 체크 {n}개", "퀵 체크 {n}개 돌려",
        "캘리브레이션 {n}개 돌려줘", "추가 데이터 {n}개 받아줘",
        "통계 더 쌓으려고 {n}개 돌려줘", "다시 {n}개 돌려줘",
        "한번 더 {n}개 받아줘", "{n}개 더 돌려",
        "cosmic 데이터 {n}개", "LED 데이터 {n}개 받아줘",
        "빔 테스트 {n}개", "잠깐 {n}개만 돌려",
        # 추가 표현
        "지금 {n}개 돌려", "바로 {n}개 받아줘", "시작해줘 {n}개",
        "{n}개 해줘", "수집 {n}개", "{n}개 수집 시작",
        "이번에 {n}개 받자", "일단 {n}개만", "{n}개 돌려볼게",
        "한판 {n}개", "{n}개짜리 돌려줘", "지금 바로 {n}개",
    ]
    for _ in range(150):
        events = _random_events()
        template = random.choice(templates)
        user_input = template.format(n=events)
        state = _make_state(random.random() > 0.3)
        decision = {
            "tool": "daq_run",
            "params": {"events": events},
            "reason": f"DAQ {events} 이벤트 수집",
        }
        examples.append(make_example(state, user_input, decision))
    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  DQM PLOT — 600 samples
#  full (explicit 200 + relative 150), heatmap (100), single (150)
# ══════════════════════════════════════════════════════════════════════════════

def _random_channel():
    tower = random.randint(1, 9)
    side = random.choice(["-C", "-S"])
    return f"T{tower}{side}"

def _random_tower_name():
    return f"T{random.randint(1, 9)}"

def _random_single_modules():
    """Return a realistic modules list for single type."""
    choice = random.random()
    if choice < 0.3:
        # one specific channel: T1-C
        return [_random_channel()]
    elif choice < 0.6:
        # tower name (auto-expands): T1
        return [_random_tower_name()]
    elif choice < 0.8:
        # scintillator + Cherenkov of same tower
        t = random.randint(1, 9)
        return [f"T{t}-S", f"T{t}-C"]
    else:
        # two different channels
        return [_random_channel(), _random_channel()]


def gen_dqm_plot() -> List[dict]:
    examples = []

    # ── A. full type — explicit run (200) ─────────────────────────────────────
    full_intadc_tmpl = [
        # bare run reference
        "run {r} 그려줘", "run {r} 플랏 보여줘", "run {r} 전부 그려줘",
        "run {r} 그래프 다 그려줘", "{r}번 런 플랏", "{r} 플랏 보여줘",
        "run {r} 그래프 보여줘", "{r}번 그려줘", "run {r} 플랏 그려",
        "{r} 데이터 그려줘", "run {r} 전체 플랏", "{r}번 런 전부 그려",
        "run {r} plot", "{r} 그래프 전부", "run {r} 다 그려",
        "run {r} DQM 그려줘", "{r}번 DQM 보여줘", "run {r} DQM plot",
        # intADC explicit
        "{r} intADC 그려줘", "run {r} intADC 보여줘",
        "{r}번 런 intADC 그려줘", "run {r} 적분 그려줘",
        "{r} 적분 ADC 보여줘", "run {r} 적분ADC 그려",
        "{r}번 intADC 플랏", "run {r} int ADC 그려줘",
        "{r} integral 그려줘", "run {r} 적분만 그려줘",
        "{r}번 런 적분 그래프", "run {r} int adc 보여줘",
        # all-tower phrasing
        "run {r} 모든 타워 intADC 그려줘", "{r}번 전체 타워 intADC 보여줘",
        "run {r} 타워 전체 그려줘", "{r} 모든 타워 그래프",
        "run {r} 모든 채널 intADC", "{r}번 전타워 플랏",
        "run {r} all tower intADC 그려", "{r} 전체 타워 그려줘",
        "run {r} 타워 다 그려", "{r}번 런 모든 타워 플랏",
    ]
    full_peakadc_tmpl = [
        "run {r} peakADC 그려줘", "run {r} peak 그려",
        "{r} peakADC 보여줘", "{r}번 peak ADC 그려줘",
        "run {r} 피크 그려", "{r} peak 플랏",
        "run {r} peakADC plot", "{r}번 런 피크ADC",
        "run {r} 피크 ADC 보여줘", "{r} peakADC",
        "run {r} peak ADC 그래프", "{r}번 peakADC 보여줘",
        "run {r} 피크만 그려줘", "{r} peakADC 그래프",
        "run {r} peak만 보여줘", "{r}번 peak 보여줘",
        "run {r} 피크ADC 그래프 보여줘", "{r} 피크 adc",
        "run {r} HV peakADC 그려줘", "{r}번 peakADC DQM",
        # all-tower phrasing
        "run {r} 모든 타워 peakADC 그려줘", "{r}번 전체 타워 peak 보여줘",
        "run {r} 타워 전체 peakADC", "{r} 모든 타워 피크 ADC",
        "run {r} all tower peakADC", "{r}번 전타워 peakADC 플랏",
    ]

    for _ in range(150):
        run = _random_run()
        tmpl = random.choice(full_intadc_tmpl)
        examples.append(make_example(
            _make_state(random.random() > 0.3),
            tmpl.format(r=run),
            {"tool": "dqm_plot",
             "params": {"run_number": run, "method": "IntADC", "type": "full"},
             "reason": f"Run {run} full IntADC DQM 플랏 생성"},
        ))

    for _ in range(50):
        run = _random_run()
        tmpl = random.choice(full_peakadc_tmpl)
        examples.append(make_example(
            _make_state(random.random() > 0.3),
            tmpl.format(r=run),
            {"tool": "dqm_plot",
             "params": {"run_number": run, "method": "PeakADC", "type": "full"},
             "reason": f"Run {run} full PeakADC DQM 플랏 생성"},
        ))

    # ── B. full type — relative reference (150) ───────────────────────────────
    relative_intadc = [
        "방금 데이터 플랏 그려줘", "이번 런 그려줘", "마지막 런 플랏 보여줘",
        "방금 받은 거 그래프", "이번 거 전부 그려줘", "지금 런 플랏 그려줘",
        "방금 거 그려줘", "이번 데이터 플랏", "마지막 데이터 그려",
        "현재 런 그래프 보여줘", "방금 돌린 거 플랏", "지금 데이터 전부 그려줘",
        "이번 런 전체 플랏 보여줘", "방금 런 plot", "최근 런 그려줘",
        "방금 데이터 그래프 보여줘", "이번 런 데이터 플랏", "지금 거 그려줘",
        "현재 데이터 그래프", "이번 런 플랏 보여줘", "마지막 거 그려줘",
        "방금 DAQ 결과 그려줘", "이번 결과 플랏 보여줘",
        "방금 거 전부 그려", "이번 런 다 그려줘", "방금 돌린 데이터 그래프",
        "현재 런 플랏 그려", "지금 런 데이터 그려줘",
        "방금 거 DQM 그려줘", "이번 런 DQM 보여줘", "방금 결과 DQM",
        "현재 런 intADC 그려줘", "이번 거 intADC 보여줘",
        "방금 런 적분 그려줘", "이번 데이터 적분 ADC",
        "방금 거 integral 그려줘", "이번 런 intADC 그래프",
        "방금 거 모든 타워 그려줘", "이번 런 전체 타워 intADC",
        "방금 런 모든 채널 그려", "이번 결과 전 타워 보여줘",
        "방금 거 타워 다 그려", "현재 런 모든 타워 플랏",
    ]
    relative_peakadc = [
        "방금 거 peakADC 그려", "이번 런 peak 그려줘", "방금 데이터 피크 보여줘",
        "마지막 런 peakADC", "지금 런 peak ADC 그려줘", "이번 거 피크 플랏",
        "방금 런 peakADC 보여", "현재 런 피크 그려줘", "방금 돌린 거 peak",
        "이번 데이터 peakADC 그려", "방금 거 피크 ADC 보여줘",
        "이번 런 피크 보여줘", "마지막 데이터 peak ADC",
        "방금 수집한 거 peakADC 그려줘", "지금 거 peak 보여줘",
        "이번 런 peakADC 그래프", "방금 DAQ 결과 피크", "현재 데이터 peak 그려",
        "방금 런 피크 그려줘", "이번 결과 peakADC 보여줘",
        "방금 거 모든 타워 peakADC", "이번 런 전체 타워 peak",
    ]

    for _ in range(110):
        tmpl = random.choice(relative_intadc)
        state = _make_state(with_agent=True)
        run = state["current_run"]
        examples.append(make_example(state, tmpl, {
            "tool": "dqm_plot",
            "params": {"run_number": run, "method": "IntADC", "type": "full"},
            "reason": f"현재 run {run} full IntADC DQM 플랏 생성",
        }))

    for _ in range(40):
        tmpl = random.choice(relative_peakadc)
        state = _make_state(with_agent=True)
        run = state["current_run"]
        examples.append(make_example(state, tmpl, {
            "tool": "dqm_plot",
            "params": {"run_number": run, "method": "PeakADC", "type": "full"},
            "reason": f"현재 run {run} full PeakADC DQM 플랏 생성",
        }))

    # ── C. heatmap type (100) ─────────────────────────────────────────────────
    heatmap_intadc_tmpl = [
        "run {r} MCPPMT intADC heatmap 그려줘",
        "run {r} heatmap IntADC 보여줘",
        "{r}번 heatmap intADC 그려",
        "run {r} MCPPMT heatmap 그려줘",
        "run {r} heatmap 그려줘",
        "{r} MCPPMT heatmap intADC",
        "run {r} heatmap intADC plot",
        "{r}번 런 MCPPMT heatmap",
        "run {r} heatmap 플랏 보여줘",
        "{r} heatmap ADC 그려",
        "run {r} MCPPMT IntADC heatmap",
        "{r}번 heatmap 플랏 IntADC",
        "run {r} heatmap 분포 그려줘",
        "{r} MCPPMT heatmap 보여줘",
        "run {r} heatmap 이미지 그려줘",
        "{r}번 런 heatmap intADC 그려줘",
        "run {r} MCPPMT 히트맵 그려줘",
        "{r} 히트맵 IntADC",
        "run {r} heatmap 분포 IntADC",
        "run {r} MCPPMT 히트맵 IntADC 보여줘",
        "run {r} MCPPMT intADC 히트맵",
        "{r}번 런 히트맵 그려줘",
        "run {r} heatmap 확인해줘",
        "{r} MCPPMT intADC heatmap 그려",
        "run {r} 히트맵 플랏",
    ]
    heatmap_peakadc_tmpl = [
        "run {r} MCPPMT peakADC heatmap 그려줘",
        "run {r} heatmap peakADC 보여줘",
        "{r}번 heatmap peakADC 그려",
        "run {r} MCPPMT heatmap peakADC",
        "{r} heatmap peak ADC",
        "run {r} heatmap peakADC plot",
        "{r}번 런 MCPPMT heatmap peakADC",
        "run {r} MCPPMT 피크 heatmap",
        "{r} heatmap 피크 ADC 그려",
        "run {r} MCPPMT peakADC 히트맵",
        "{r}번 히트맵 peakADC",
        "run {r} heatmap 피크 그려줘",
        "{r}번 heatmap peak 그려줘",
        "run {r} MCPPMT 히트맵 peakADC 보여줘",
        "run {r} heatmap PeakADC",
    ]

    for _ in range(65):
        run = _random_run()
        tmpl = random.choice(heatmap_intadc_tmpl)
        examples.append(make_example(
            _make_state(random.random() > 0.3),
            tmpl.format(r=run),
            {"tool": "dqm_plot",
             "params": {"run_number": run, "method": "IntADC",
                        "type": "heatmap", "modules": ["MCPPMT"]},
             "reason": f"Run {run} heatmap IntADC (MCPPMT) DQM 플랏 생성"},
        ))

    for _ in range(35):
        run = _random_run()
        tmpl = random.choice(heatmap_peakadc_tmpl)
        examples.append(make_example(
            _make_state(random.random() > 0.3),
            tmpl.format(r=run),
            {"tool": "dqm_plot",
             "params": {"run_number": run, "method": "PeakADC",
                        "type": "heatmap", "modules": ["MCPPMT"]},
             "reason": f"Run {run} heatmap PeakADC (MCPPMT) DQM 플랏 생성"},
        ))

    # ── D. single type — specific tower or channel (150) ─────────────────────
    single_intadc_tmpl = [
        # channel name patterns
        "run {r} {ch} intADC 그려줘",
        "run {r} {ch} IntADC 보여줘",
        "{r}번 {ch} intADC 그려",
        "run {r} {ch} intADC plot",
        "{r} {ch} intADC 플랏",
        "run {r} {ch} 적분 ADC 그려줘",
        "{r}번 런 {ch} intADC",
        "run {r} {ch} integral ADC 그려줘",
        "{r} {ch} intADC 보여줘",
        "run {r} {ch} 파형 적분 그려",
        "run {r} {ch} intADC 그래프",
        "{r}번 {ch} 적분 그려줘",
        "run {r} {ch} channel intADC",
        "{r} {ch} 채널 intADC 그려",
        "run {r} {ch}만 intADC 그려줘",
        "run {r} {ch} intADC 확인해줘",
        "{r}번 런 {ch} intADC 확인",
        "run {r} {ch} 그래프 보여줘",
        "{r} {ch} 그려줘",
        "run {r} {ch} 플랏",
    ]
    single_peakadc_tmpl = [
        "run {r} {ch} peakADC 그려줘",
        "run {r} {ch} PeakADC 보여줘",
        "{r}번 {ch} peakADC 그려",
        "run {r} {ch} peak ADC 그려줘",
        "{r} {ch} peakADC 플랏",
        "run {r} {ch} 피크 ADC 그려줘",
        "{r}번 런 {ch} peakADC",
        "run {r} {ch} peakADC plot",
        "{r} {ch} peak 보여줘",
        "run {r} {ch} peakADC 그래프",
        "{r}번 {ch} 피크 그려줘",
        "run {r} {ch}만 peakADC 그려줘",
        "run {r} {ch} peak 확인해줘",
        "{r}번 {ch} PeakADC 확인",
        "run {r} {ch} 피크 플랏",
    ]

    for _ in range(100):
        run = _random_run()
        modules = _random_single_modules()
        ch_str = " ".join(modules)   # e.g. "T1-C" or "T1" or "T3-S T3-C"
        tmpl = random.choice(single_intadc_tmpl)
        examples.append(make_example(
            _make_state(random.random() > 0.3),
            tmpl.format(r=run, ch=ch_str),
            {"tool": "dqm_plot",
             "params": {"run_number": run, "method": "IntADC",
                        "type": "single", "modules": modules},
             "reason": f"Run {run} single IntADC ({ch_str}) DQM 플랏 생성"},
        ))

    for _ in range(50):
        run = _random_run()
        modules = _random_single_modules()
        ch_str = " ".join(modules)
        tmpl = random.choice(single_peakadc_tmpl)
        examples.append(make_example(
            _make_state(random.random() > 0.3),
            tmpl.format(r=run, ch=ch_str),
            {"tool": "dqm_plot",
             "params": {"run_number": run, "method": "PeakADC",
                        "type": "single", "modules": modules},
             "reason": f"Run {run} single PeakADC ({ch_str}) DQM 플랏 생성"},
        ))

    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  RUN LOG — 100 samples (70 explicit + 30 relative)
# ══════════════════════════════════════════════════════════════════════════════

def gen_run_log() -> List[dict]:
    examples = []

    columns_values = {
        "program": ["EM", "Calib", "HV_EQ", "Pedestal", "LED", "test", "cosmic",
                     "beam_test", "noise", "threshold_scan"],
        "notes": ["pedestal 불안정", "beam unstable", "좋은 데이터", "HV trip 발생",
                   "trigger rate 낮음", "재측정 필요", "테스트 런", "빔 불안정",
                   "DAQ error 발생", "좋은 품질", "detector noise 높음",
                   "energy scan 완료", "position 확인 필요", "HV 안정적",
                   "빔 꺼짐", "spill 불안정", "dark count 높음", "좋음",
                   "beam 정상", "타겟 변경", "trigger 조정 필요"],
        "config": ["standard", "high_gain", "low_threshold", "test_config",
                    "noise_run", "default", "calibration", "debug"],
        "beam_energy": ["1", "2", "3", "5", "10", "20", "30", "40", "60", "80", "100", "120"],
        "beam_type": ["electron", "pion", "muon", "proton", "positron"],
        "trigger_setup": ["standard", "prescale_10", "random", "external",
                          "self_trigger", "cosmic_trigger", "LED_trigger"],
        "hv_drc": ["ON", "OFF", "1400V", "1500V", "1550V", "1600V", "1650V", "1700V"],
        "hv_aux": ["ON", "OFF", "750V", "800V", "850V", "900V", "950V", "1000V"],
    }

    col_korean = {
        "program": ["프로그램", "program", "프로그램을", "프로그램에", "프로그램 란에"],
        "notes": ["노트", "메모", "비고", "노트에", "메모에", "비고에", "메모란에", "notes"],
        "config": ["설정", "config", "설정을", "설정에", "설정란에"],
        "beam_energy": ["빔 에너지", "에너지", "빔에너지를", "에너지를", "beam energy를"],
        "beam_type": ["빔 타입", "타입", "빔타입을", "타입을", "beam type을"],
        "trigger_setup": ["트리거", "trigger", "트리거를", "트리거 설정", "트리거에"],
        "hv_drc": ["HV DRC", "drc", "DRC를", "DRC에", "HV DRC를"],
        "hv_aux": ["HV Aux", "aux", "Aux를", "Aux에", "HV Aux를"],
    }

    explicit_templates = [
        # 추가/쓰기
        "run {r} {col_kr} {val} 추가해줘",  "run {r} {col_kr} {val} 써줘",
        "run {r} {col_kr} {val} 기록해줘",  "run {r} {col_kr} {val} 넣어줘",
        "run {r} {col_kr} {val} 입력해줘",  "run {r} {col_kr} {val} 적어줘",
        "run {r} 로그 {col_kr} {val} 추가", "{r}번 런 로그 {col_kr} {val}",
        "{r} {col_kr} {val} 추가",          "{r}번 {col_kr} {val} 써줘",
        # 수정/변경
        "run {r} {col_kr} {val}로 수정해줘",   "run {r} {col_kr} {val}로 바꿔줘",
        "run {r} {col_kr} {val}로 업데이트",    "run {r} {col_kr} {val}로 변경해줘",
        "run {r} {col_kr} {val}로 고쳐줘",      "{r}번 런 {col_kr} {val}로 바꿔줘",
        "{r} {col_kr} {val}로 수정",            "{r}번 {col_kr} {val}로 변경",
        "{r} 로그 {col_kr} {val}로 업데이트해줘",
        # 축약
        "{r} {col_kr} {val}",  "run {r} {col_kr} {val}",  "{r}번 {col_kr} {val}",
        "{r} 로그 {col_kr} {val}",
        # 존댓말
        "run {r} {col_kr} {val}로 수정해주세요",   "{r}번 {col_kr} {val}로 바꿔주세요",
        "run {r} {col_kr} {val} 기록해주세요",
        # 기타
        "run {r} {col_kr} {val}로 해줘",  "{r} {col_kr} {val}이야",
        "run {r} {col_kr} {val}임",
    ]

    for _ in range(100):
        run = _random_run()
        col = random.choice(list(columns_values.keys()))
        val = random.choice(columns_values[col])
        col_kr = random.choice(col_korean[col])
        template = random.choice(explicit_templates)
        user_input = template.format(r=run, col_kr=col_kr, val=val)
        state = _make_state(random.random() > 0.3)
        decision = {
            "tool": "run_log",
            "params": {"command": "update", "run_num": run, col: val},
            "reason": f"Run {run} {col} 업데이트",
        }
        examples.append(make_example(state, user_input, decision))

    # Relative (50)
    relative_templates = [
        "방금 런 {col_kr} {val} 추가해줘",       "이번 런 {col_kr} {val}로 수정",
        "지금 런 {col_kr} {val} 기록해줘",        "방금 거 {col_kr} {val}로 바꿔줘",
        "이번 데이터 {col_kr} {val}",             "현재 런 {col_kr} {val} 써줘",
        "마지막 런 {col_kr} {val} 추가",          "방금 돌린 런 {col_kr} {val}로 업데이트",
        "이번 거 로그 {col_kr} {val}",            "방금 거 {col_kr} {val} 넣어줘",
        "이번 런 {col_kr} {val}로 변경",          "방금 런 {col_kr} {val}로 해줘",
        "현재 런 {col_kr} {val}로 수정해줘",      "지금 거 {col_kr} {val} 추가",
        "방금 거 {col_kr} {val}",
        # 추가 패턴
        "이번 결과 {col_kr} {val}로 저장",        "방금 돌린 거 {col_kr} {val}로 바꿔",
        "최근 런 {col_kr} {val}",                  "이거 {col_kr} {val}로 해줘",
        "방금 수집한 거 {col_kr} {val} 기록",
    ]
    for _ in range(50):
        col = random.choice(list(columns_values.keys()))
        val = random.choice(columns_values[col])
        col_kr = random.choice(col_korean[col])
        template = random.choice(relative_templates)
        user_input = template.format(col_kr=col_kr, val=val)
        state = _make_state(with_agent=True)
        decision = {
            "tool": "run_log",
            "params": {"command": "update", "run_num": state["current_run"], col: val},
            "reason": f"현재 run {state['current_run']} {col} 업데이트",
        }
        examples.append(make_example(state, user_input, decision))

    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  RUN LOG READ — 50 samples (run_log의 100에 포함)
# ══════════════════════════════════════════════════════════════════════════════

def gen_run_log_read() -> List[dict]:
    """Log read requests — user wants to VIEW log, not modify."""
    examples = []

    # Explicit run number
    explicit = [
        "run {r} 로그 확인해줘", "run {r} 로그 보여줘", "{r}번 로그 확인",
        "{r} 로그 열어줘", "run {r} 기록 보여줘", "{r}번 런 로그 확인",
        "run {r} 로그 조회", "{r} 로그 읽어줘", "run {r} 로그 내용",
        "{r}번 런 기록 확인해줘", "run {r} log 보여줘", "run {r} 로그 알려줘",
        "{r}번 런 로그 보여줘", "run {r} 정보 확인", "{r} 로그 정보",
        "run {r} 런 정보 보여줘", "{r}번 런 확인해줘", "run {r} 로그 내용 보여줘",
        "{r} 기록 확인", "run {r} 기록 알려줘",
    ]
    for _ in range(40):
        r = _random_run()
        template = random.choice(explicit)
        user_input = template.format(r=r)
        state = _make_state(random.random() > 0.3)
        decision = {
            "tool": "run_log",
            "params": {"command": "read", "run_num": r},
            "reason": f"Run {r} 로그 조회",
        }
        examples.append(make_example(state, user_input, decision))

    # Relative
    relative = [
        "방금 런 로그 확인해줘", "이번 런 로그 보여줘", "지금 런 로그 확인",
        "마지막 런 로그 보여줘", "방금 런 기록 확인", "이번 런 기록 보여줘",
        "현재 런 로그 알려줘", "방금 거 로그 확인", "이번 거 로그 보여줘",
        "방금 돌린 런 로그", "마지막 런 정보 확인", "이번 런 정보 보여줘",
        "방금 런 로그 알려줘", "현재 런 기록 확인해줘",
        # 추가
        "이번 결과 로그 보여줘", "방금 거 기록 확인", "지금 런 정보",
        "최근 런 로그 조회", "이번 런 내용 확인",
    ]
    for _ in range(30):
        template = random.choice(relative)
        state = _make_state(with_agent=True)
        decision = {
            "tool": "run_log",
            "params": {"command": "read", "run_num": state["current_run"]},
            "reason": f"현재 run {state['current_run']} 로그 조회",
        }
        examples.append(make_example(state, template, decision))

    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  HV READ — 100 samples
# ══════════════════════════════════════════════════════════════════════════════

def gen_hv_read() -> List[dict]:
    examples = []
    templates = [
        # 기본
        "HV 값 읽어줘", "지금 HV 얼마야", "현재 고전압 확인해줘",
        "HV config 보여줘", "전압 값 확인", "HV 상태 알려줘",
        "고전압 얼마로 설정되어 있어", "HV 읽어", "HV 전압 읽어줘",
        "고전압 값 확인",
        # 존댓말
        "HV 값 확인해주세요", "현재 전압 좀 읽어주세요", "고전압 상태 알려주세요",
        "HV 설정값 확인 부탁드립니다", "전압 값 좀 보여주세요",
        "HV 얼마인지 확인해주세요", "고전압 읽어주세요",
        # 반말
        "HV 얼마야", "전압 얼마", "HV 확인", "고전압 읽어",
        "HV 좀 봐줘", "전압 확인해봐", "HV 좀 알려줘", "전압 좀",
        "HV 봐봐", "고전압 좀 봐",
        # 영어
        "read HV config", "HV status", "check HV values", "show HV config",
        "HV read", "read HV", "get HV values", "HV check", "show HV",
        "HV config read",
        # 구체적
        "지금 타워 HV 얼마야", "현재 HV 전압 상태", "HV 전압 다 보여줘",
        "CAEN HV 값 읽어줘", "고전압 모듈 상태 확인", "HV 세팅 보여줘",
        "전체 HV 값 확인해줘", "DRC HV 얼마야", "Aux HV 값 알려줘",
        "HV 모듈 전압 확인", "전체 고전압 세팅", "HV 값 전부 보여줘",
        # 질문형
        "HV가 지금 얼마로 되어있어?", "전압이 몇이야?",
        "고전압 설정 어떻게 되어있어?", "현재 HV 설정은?",
        "HV 값이 뭐야?", "전압 세팅 뭐로 되어있어?",
        "지금 전압 세팅 알려줘", "HV configuration 보여줘",
        "전압 config 읽어줘", "현재 고전압 세팅값",
        "HV 뭐로 세팅되어있어?", "전압값 몇이지?", "고전압이 얼마야?",
        "HV 전압 몇 볼트야?", "지금 HV config가 뭐야?",
        # 추가
        "채널별 HV 알려줘", "T1 HV 얼마야", "T5 전압 확인",
        "DRC 채널 HV 확인", "지금 세팅된 전압 다 보여줘",
        "HV 모두 확인", "전체 채널 전압 상태", "HV 지금 어떻게 돼있어",
    ]
    for _ in range(150):
        template = random.choice(templates)
        state = _make_state(random.random() > 0.3)
        decision = {
            "tool": "hv_read",
            "params": {"command": "status", "channels": "all"},
            "reason": "현재 HV 상태 읽기",
        }
        examples.append(make_example(state, template, decision))
    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  HV WRITE — 전압 변경, ON/OFF (BrainAgent가 확인 후 실행)
#  ※ BrainAgent의 handle_request가 실행 전 팝업에서 사용자 확인을 받음
# ══════════════════════════════════════════════════════════════════════════════

def _parse_voltage(v_str: str) -> float:
    """Strip 'V' suffix and convert to float."""
    return float(v_str.rstrip("Vv").strip())


def gen_hv_write() -> List[dict]:
    """HV write requests — voltage change, on/off. BrainAgent will confirm before executing."""
    examples = []

    channels = ["T1C", "T1S", "T2C", "T2S", "T3C", "T3S", "T4C", "T4S",
                "T5C", "T5S", "T6C", "T6S", "T7C", "T7S", "T8C", "T8S", "T9C", "T9S"]
    # Include low values (test/pedestal) and standard HV values; some without V suffix
    voltages_with_v  = ["100V", "200V", "500V", "800V", "900V", "1000V",
                        "1200V", "1400V", "1500V", "1550V", "1600V", "1650V", "1700V"]
    voltages_no_v    = ["100", "200", "500", "800", "900", "1000",
                        "1200", "1400", "1500", "1550", "1600", "1650", "1700"]

    def _rand_volt():
        """Return (display_string, numeric_value) with or without V suffix."""
        if random.random() < 0.5:
            s = random.choice(voltages_with_v)
        else:
            s = random.choice(voltages_no_v)
        return s, _parse_voltage(s)

    # ── 단일 채널 전압 변경 (120 samples) ──
    # Critically: include "전압" keyword between channel and value
    templates_ch_voltage = [
        # 동사 있음
        "{ch} {v}로 수정", "{ch} {v}로 바꿔줘", "{ch} 전압 {v}로 변경",
        "{ch}를 {v}로 설정", "{ch} {v}로 올려줘", "{ch} {v}로 내려줘",
        "HV {ch} {v}로 수정", "HV {ch} {v}로 바꿔",
        "{ch} 전압을 {v}로 해줘", "{ch} {v} 설정해줘", "{ch} {v}로 맞춰줘",
        "HV {ch} {v} 세팅", "{ch} 전압 {v}로 조정", "{ch} {v} 적용해줘",
        "{ch} 전압 {v}로 바꿔줘", "{ch} 전압을 {v}로 변경해줘",
        "{ch} 전압 {v}로 설정해줘", "{ch} 전압 {v}로 맞춰줘",
        "{ch} HV {v}로 올려줘", "{ch} 고전압 {v}로 설정",
        "{ch} 전압 {v}로 올려줘", "{ch} 전압 {v}로 내려줘",
        "{ch} {v}로 인가해줘", "{ch} {v}로 해줘",
        # 동사 없음 (bare) — 가장 오해 잦은 패턴
        "{ch} 전압 {v}으로", "{ch} 전압 {v}로",
        "{ch} 전압 {v}", "{ch} {v}",
        "{ch} 전압 {v}으로 해", "{ch} {v}으로 해",
        # 존댓말
        "{ch} {v}로 수정해주세요", "{ch} 전압 {v}로 바꿔주세요",
        "{ch} 전압을 {v}로 설정해주세요", "{ch} {v}로 변경 부탁드립니다",
        # 영어 혼용
        "{ch} set to {v}", "{ch} voltage {v}",
        "set {ch} to {v}", "{ch} {v} please",
    ]
    for _ in range(120):
        ch = random.choice(channels)
        v_str, v_num = _rand_volt()
        template = random.choice(templates_ch_voltage)
        user_input = template.format(ch=ch, v=v_str)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, user_input, {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": [ch], "voltage": v_num},
            "reason": f"{ch} 채널 전압을 {v_num}V로 변경",
        }))

    # ── 여러 채널 동시 변경 (30 samples) ──
    templates_multi = [
        "{ch1}, {ch2} {v}로 수정", "{ch1}와 {ch2} 전압 {v}로",
        "{ch1} {ch2} {v}로 설정", "{ch1},{ch2} {v}로 바꿔줘",
        "{ch1} {ch2} 전압 {v}로 변경", "{ch1}, {ch2} 전압 {v}",
        "{ch1},{ch2} 전압 {v}로 맞춰줘",
        "{ch1} 및 {ch2} {v}로 수정", "{ch1}와 {ch2} {v}로 바꿔",
        "{ch1} {ch2} 전압 {v}으로",
    ]
    for _ in range(30):
        ch1, ch2 = random.sample(channels, 2)
        v_str, v_num = _rand_volt()
        template = random.choice(templates_multi)
        user_input = template.format(ch1=ch1, ch2=ch2, v=v_str)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, user_input, {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": [ch1, ch2], "voltage": v_num},
            "reason": f"{ch1}, {ch2} 채널 전압을 {v_num}V로 변경",
        }))

    # ── 전체 채널 변경 — ALL 키워드 필수 (30 samples) ──
    # Must include "전체", "모든", "전 채널", "HV" with explicit all-scope keyword
    templates_all = [
        "HV 전체 {v}로 수정", "모든 채널 {v}로 변경", "전체 HV {v}로 설정",
        "전 채널 {v}로 바꿔줘", "전체 전압 {v}로 맞춰줘",
        "모든 HV {v}로", "전체 채널 전압 {v}로 수정",
        "HV 전체 {v}로 바꿔줘", "전 채널 전압 {v}로",
        "모든 채널 전압 {v}로 변경해줘", "HV 전부 {v}로 설정",
        "전체 HV 전압 {v}로 변경", "모든 타워 HV {v}로 수정",
        "전채널 {v}로 바꿔", "전체 고전압 {v}로",
    ]
    for _ in range(30):
        v_str, v_num = _rand_volt()
        template = random.choice(templates_all)
        user_input = template.format(v=v_str)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, user_input, {
            "tool": "hv_write",
            "params": {"command": "voltage", "channels": "all", "voltage": v_num},
            "reason": f"모든 채널 전압을 {v_num}V로 변경",
        }))

    # ── ON/OFF (20 samples) ──
    templates_on = [
        "HV 켜줘", "HV 전원 켜", "HV on", "HV turn on", "전체 HV 켜줘",
        "HV 다시 켜줘", "모든 HV 켜", "HV 전체 켜",
        "고전압 켜줘", "HV 전원 올려줘", "HV 올려",
    ]
    for _ in range(10):
        template = random.choice(templates_on)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, template, {
            "tool": "hv_write",
            "params": {"command": "on", "channels": "all"},
            "reason": "모든 HV 채널 켜기",
        }))

    templates_off = [
        "HV 꺼줘", "HV 전원 꺼", "HV off", "HV turn off", "전체 HV 꺼줘",
        "HV 다 꺼줘", "모든 HV 꺼", "HV 전체 꺼",
        "고전압 꺼줘", "HV 전원 내려줘", "HV 내려",
    ]
    for _ in range(10):
        template = random.choice(templates_off)
        state = _make_state(random.random() > 0.3)
        examples.append(make_example(state, template, {
            "tool": "hv_write",
            "params": {"command": "off", "channels": "all"},
            "reason": "모든 HV 채널 끄기",
        }))

    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  NONE (unclear / log-view / clarification) — 100 samples
# ══════════════════════════════════════════════════════════════════════════════

def gen_unclear() -> List[dict]:
    examples = []

    templates = [
        # ── 로그 조회 — 런 번호 없음 ──
        ("로그 확인", "어떤 런 번호의 로그를 확인할까요?"),
        ("로그 보여줘", "어떤 런 번호의 로그를 볼까요?"),
        ("로그 열어줘", "어떤 런 번호의 로그를 확인할까요?"),
        ("로그 알려줘", "어떤 런 번호의 로그를 확인할까요?"),
        ("기록 확인", "어떤 런 번호의 기록을 확인할까요?"),

        # ── 로그 수정 불명확 (열+값 부족) ──
        ("로그 수정해줘", "어떤 런 번호의 어느 열을 수정할까요?"),
        ("로그 추가해줘", "어떤 런 번호의 어느 열에 무엇을 추가할까요?"),
        ("로그 업데이트", "어떤 런 번호의 어느 열을 수정할까요?"),
        ("기록해줘", "어떤 런 번호의 어느 열에 기록할까요?"),
        ("로그에 써줘", "어떤 런 번호의 어느 열에 무엇을 쓸까요?"),
        ("log update", "어떤 런 번호의 어느 열을 수정할까요?"),
        ("run {r} 프로그램 수정해줘", "Run {r} 프로그램을 무엇으로 바꿀까요?"),
        ("run {r} 메모 추가해줘", "Run {r} 메모에 무엇을 쓸까요?"),
        ("run {r} 노트 바꿔줘", "Run {r} 노트를 무엇으로 바꿀까요?"),
        ("{r} 에너지 수정", "Run {r} 빔 에너지를 얼마로 수정할까요?"),
        ("run {r} 설정 바꿔줘", "Run {r} 설정을 무엇으로 바꿀까요?"),
        ("방금 런 프로그램 수정", "프로그램을 무엇으로 바꿀까요?"),

        # ── DQM 플랏 불명확 (런 번호 누락) ──
        ("그려줘", "어떤 런 번호의 DQM 플랏을 그릴까요?"),
        ("플랏 보여줘", "어떤 런 번호의 DQM 플랏을 그릴까요?"),
        ("그래프", "어떤 런 번호의 그래프를 그릴까요?"),
        ("plot", "어떤 런 번호의 DQM 플랏을 그릴까요?"),
        ("플랏", "어떤 런 번호의 DQM 플랏을 그릴까요?"),
        ("그래프 보여줘", "어떤 런 번호의 그래프를 그릴까요?"),
        ("그림 그려줘", "어떤 런 번호의 DQM 플랏을 그릴까요?"),
        ("DQM 그려줘", "어떤 런 번호의 DQM 플랏을 그릴까요?"),
        ("peakADC 보여줘", "어떤 런 번호의 DQM peakADC를 그릴까요?"),
        ("intADC 보여줘", "어떤 런 번호의 DQM intADC를 그릴까요?"),

        # ── DAQ 불명확 (이벤트 수 누락) ──
        ("DAQ 돌려줘", "몇 개의 이벤트를 수집할까요?"),
        ("데이터 받아줘", "몇 개의 이벤트를 수집할까요?"),
        ("이벤트 받아줘", "몇 개의 이벤트를 수집할까요?"),
        ("돌려줘", "DAQ를 몇 개 이벤트로 돌릴까요?"),
        ("DAQ run", "몇 개의 이벤트를 수집할까요?"),
        ("데이터 수집해줘", "몇 개의 이벤트를 수집할까요?"),
        ("DAQ 한번 돌려줘", "몇 개의 이벤트를 수집할까요?"),
        ("데이터 좀 받자", "몇 개의 이벤트를 수집할까요?"),
        ("이벤트 수집", "몇 개의 이벤트를 수집할까요?"),

        # ── 런 번호만 ──
        ("12905", "Run 12905에 대해 무엇을 할까요? (플랏, 로그 수정 등)"),
        ("run 12905", "Run 12905에 대해 무엇을 할까요? (플랏, 로그 수정 등)"),
        ("12905번", "Run 12905에 대해 무엇을 할까요? (플랏, 로그 수정 등)"),

        # ── 일반 불명확 ──
        ("확인해줘", "무엇을 확인할까요? (플랏, HV, 로그 등)"),
        ("도와줘", "무엇을 도와드릴까요? DAQ 실행, 플랏 생성, 로그 수정 등이 가능합니다."),
        ("데이터", "데이터를 수집할까요, 아니면 기존 데이터의 플랏을 그릴까요?"),
        ("뭐 할 수 있어?", "DAQ 실행, 플랏 생성, 로그 수정, HV 값 읽기가 가능합니다."),
        ("help", "DAQ 실행, 플랏 생성, 로그 수정, HV 값 읽기가 가능합니다."),
        ("도움말", "DAQ 실행, 플랏 생성, 로그 수정, HV 값 읽기가 가능합니다."),
        ("뭐 할 수 있어", "DAQ 실행, 플랏 생성, 로그 수정, HV 값 읽기가 가능합니다."),
        ("명령어 목록", "DAQ 실행, 플랏 생성, 로그 수정, HV 값 읽기가 가능합니다."),

        # ── 의미 불분명 ──
        ("이상한데", "무엇이 이상한가요? 구체적으로 알려주시면 도움을 드릴 수 있습니다."),
        ("잘 안 돼", "무엇이 안 되나요? 구체적으로 알려주시면 도움을 드릴 수 있습니다."),
        ("다시 해줘", "무엇을 다시 할까요? (DAQ, 플랏, 로그 등)"),
        ("다시", "무엇을 다시 할까요? (DAQ, 플랏, 로그 등)"),
        ("한번 더", "무엇을 한번 더 할까요? (DAQ, 플랏, 로그 등)"),
        ("이거 뭐야", "무엇에 대해 알고 싶으신가요?"),
        ("상태", "무엇의 상태를 확인할까요? (DAQ, HV 등)"),

        # ── 잡담/인사 ──
        ("ㅎㅎ", "무엇을 도와드릴까요?"),
        ("ㅋㅋ", "무엇을 도와드릴까요?"),
        ("감사", "무엇을 더 도와드릴까요?"),
        ("고마워", "무엇을 더 도와드릴까요?"),
        ("ok", "무엇을 더 도와드릴까요?"),
        ("ㅇㅇ", "무엇을 도와드릴까요?"),
        ("안녕", "무엇을 도와드릴까요? DAQ 실행, 플랏 생성, 로그 수정 등이 가능합니다."),
        ("수고", "무엇을 더 도와드릴까요?"),
    ]

    for _ in range(150):
        user_input, message = random.choice(templates)
        r = _random_run()
        if "{r}" in user_input:
            user_input = user_input.replace("{r}", str(r))
            message = message.replace("{r}", str(r))
        if "12905" in user_input:
            user_input = user_input.replace("12905", str(r))
            message = message.replace("12905", str(r))
        state = _make_state(random.random() > 0.5)
        decision = {"tool": "none", "message": message}
        examples.append(make_example(state, user_input, decision))

    return examples


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    output_file = Path(__file__).parent / "data" / "brain_data.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_examples = []
    all_examples.extend(gen_daq_run())            # 150
    all_examples.extend(gen_dqm_plot())           # 600  (full 350 + heatmap 100 + single 150)
    all_examples.extend(gen_run_log())            # 150  (write)
    all_examples.extend(gen_run_log_read())       #  70  (read)
    all_examples.extend(gen_hv_read())            # 150
    all_examples.extend(gen_hv_write())           # 210  (single 120 + multi 30 + all 30 + on/off 20+10)
    all_examples.extend(gen_unclear())            # 150

    random.shuffle(all_examples)

    with open(output_file, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    lengths = [sum(len(m["content"]) for m in ex["messages"]) for ex in all_examples]
    max_chars = max(lengths)
    avg_chars = sum(lengths) / len(lengths)
    print(f"Generated {len(all_examples)} samples -> {output_file}")
    print(f"   char len  max={max_chars:,}  avg={avg_chars:,.0f}  "
          f"(≈token max={max_chars // 2:,}  avg={avg_chars // 2:,.0f})")


if __name__ == "__main__":
    main()

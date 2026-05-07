#!/usr/bin/env python3
"""
Training data generator for Calibration Scan Agent
- build_full_context, _build_state_context, _get_step_hint 포맷이
  calib_scan_agent.py와 완전히 동일하도록 유지
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

TOWER_ORDER = ["T1", "T2", "T3", "T6", "T5", "T4", "T7", "T8", "T9"]

# Workflow 메시지 템플릿 (agent와 동일)
MESSAGE_ENERGY_REQ = "에너지를 입력하세요."
MESSAGE_EVENTS_REQ = "이벤트를 몇개 받을까요?"
MESSAGE_MOVE_REQ = "{tower} 타워 중심으로 이동해주세요 (x:{x}, y:{y})."
MESSAGE_PLOT_CONFIRM = "데이터 수집 및 Plot 생성이 완료되었습니다. 결과를 확인해주세요."
MESSAGE_COMPLETE = "모든 타워에 대한 스캔이 완료되었습니다."

SYSTEM_PROMPT = """You are Calibration Scan Agent for test beam experiments.

Follow these steps EXACTLY:

=== STEP 0: Configuration ===
- If 'beam_energy' is null: Request energy from user.
  Output: {"message": "에너지를 입력하세요."}
- After energy is provided: Update 'beam_energy' and request event count.
  CRITICAL: beam_energy is in GeV. Store the number exactly as user inputs. (e.g. user inputs "2" → beam_energy: 2, NOT 2000)
  Output: {"message": "이벤트를 몇개 받을까요?", "update_state": {"beam_energy": <number in GeV>, "phase": "config_events"}}
- After event count is provided: Update 'target_events' and set phase to 'idle'.
  Output: {"tool": "none", "update_state": {"target_events": <number>, "phase": "idle"}}

=== STEP 1: For Each Tower in tower_order (REPEAT for T1-T9) ===
Repeat steps 1a-1c for each tower in tower_order until all towers are completed.

1a. Request Tower Movement
Output: {"message": "{tower} 타워 중심으로 이동해주세요 (x:{x}, y:{y}).", "update_state": {"current_tower": "{tower}"}}
(Note: replace {tower}, {x}, {y} with values from state)

After user says "완료":
CRITICAL: When you see "완료" in conversation history, proceed to next step immediately.
1b. Execute DAQ
Tool: "daq_run_tool"
Params: {
    "events": <target_events from state>,
    "config": "setup",
    "pos_h": <x>,
    "pos_v": <y>,
    "pos_rot": 0.0,
    "pos_tilt": 0.0,
    "beam_energy": <beam_energy from state>
}
(Plot is auto-rendered by DQM live during DAQ — never call any plot tool.)

1c. Request Plot Confirmation
Output: {"message": "데이터 수집 및 Plot 생성이 완료되었습니다. 결과를 확인해주세요."}

After user says "완료":
CRITICAL: When you see "완료" in conversation history, you MUST mark current tower as completed.
Output: {"tool": "none", "update_state": {"tower_status": {"{tower}": {"completed": true}}}}
(Note: replace {tower} with the CURRENT tower name shown in state. Do NOT include current_tower_idx.)

=== STEP 2: Completion ===
When ALL towers are completed, output:
{"message": "모든 타워에 대한 스캔이 완료되었습니다."}

=== CRITICAL RULES ===
1. Follow steps STRICTLY in order. Do NOT skip or reorder steps.
2. Use EXACT messages above. DO NOT change or paraphrase.
3. Output JSON format (CHOOSE ONE, NEVER BOTH):
   - {"tool": "...", "params": {...}, "update_state": {...}}
   - {"message": "...", "update_state": {...}}
   - {"tool": "none", "update_state": {...}}
4. MOST IMPORTANT: When you see "완료" in conversation history, you MUST process it IMMEDIATELY.
"""


def _build_state_context(state: Dict, tower_positions: Dict) -> str:
    """agent의 _build_state_context()와 동일"""
    lines = []
    lines.append(f"Phase: {state['phase']}")
    lines.append(f"Beam Energy: {state['beam_energy']} GeV")
    lines.append(f"Target Events: {state['target_events']}")
    lines.append("")
    lines.append("Tower Progress:")
    for i, tower in enumerate(TOWER_ORDER):
        status = state['tower_status'][tower]
        pos = tower_positions.get(tower, {'x': 0, 'y': 0})
        if status['completed']:
            lines.append(f"  ✅ {tower} (x:{pos['x']:.1f}, y:{pos['y']:.1f}): Completed (Runs: {status['runs']})")
        elif i == state['current_tower_idx']:
            lines.append(f"  ➡️  {tower} (x:{pos['x']:.1f}, y:{pos['y']:.1f}): Pending  <- CURRENT (target: {state['target_events']} events)")
        else:
            lines.append(f"     {tower} (x:{pos['x']:.1f}, y:{pos['y']:.1f}): Pending")
    return "\n".join(lines)


def _build_history_context(history: List[Dict]) -> str:
    """agent의 build_full_context() 내 history 처리와 동일 (10턴, 완료 하이라이팅)"""
    if not history:
        return "(No conversation yet)"
    lines = []
    for msg in history[-10:]:
        role = "User" if msg["role"] == "user" else "Agent"
        content = msg["content"]
        if content == "완료" or content.strip() == "완료":
            lines.append(f"{role}: 완료 [IMPORTANT: User confirmed completion]")
        else:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _get_step_hint(state: Dict, history: List[Dict]) -> str:
    """agent의 _get_step_hint()와 동일 - 상태 요약만, AI가 스스로 결정"""
    phase = state.get("phase", "config")
    tower_idx = state.get("current_tower_idx", 0)
    current_tower = TOWER_ORDER[tower_idx] if tower_idx < len(TOWER_ORDER) else "All completed"
    total = len(TOWER_ORDER)
    return f"Phase: {phase} | Tower: {current_tower} ({tower_idx + 1}/{total})"


def build_full_context(state: Dict, history: List[Dict], tower_positions: Dict,
                       current_input: Optional[str] = None) -> str:
    """agent의 build_full_context()와 동일한 구조"""
    # 마지막 user 메시지를 current_input으로 분리
    if current_input is None and history and history[-1]["role"] == "user":
        current_input = history[-1]["content"]
        temp_history = history[:-1]
    else:
        temp_history = history

    parts = []
    parts.append("=== Current State ===")
    parts.append(_build_state_context(state, tower_positions))
    parts.append("")
    parts.append("=== Recent Conversation ===")
    parts.append(_build_history_context(temp_history))
    parts.append("")
    if current_input:
        parts.append("=== Current User Input ===")
        parts.append(current_input)
        parts.append("")
    parts.append("=== Your Task ===")
    parts.append(_get_step_hint(state, history))
    parts.append("")
    parts.append("Output JSON with tool name and parameters.")
    return "\n".join(parts)


def make_example(state, history, tower_positions, decision, current_input=None):
    ctx = build_full_context(state, history, tower_positions, current_input)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ctx},
            {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)}
        ]
    }


def generate_workflow_normal(energy: float, events: int) -> List[Dict[str, Any]]:
    examples = []
    history = []
    tower_positions = {t: {'x': round(random.uniform(500, 5000), 1),
                           'y': round(random.uniform(500, 5000), 1)} for t in TOWER_ORDER}
    state = {
        "phase": "config",
        "beam_energy": None,
        "target_events": None,
        "current_tower_idx": 0,
        "tower_status": {t: {"collected_events": 0, "runs": [], "completed": False} for t in TOWER_ORDER}
    }

    # --- STEP 0 ---
    dec = {"message": MESSAGE_ENERGY_REQ}
    examples.append(make_example(state, history, tower_positions, dec))
    history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
    history.append({"role": "user", "content": str(int(energy))})
    state["beam_energy"] = energy

    dec = {"message": MESSAGE_EVENTS_REQ, "update_state": {"beam_energy": energy, "phase": "config_events"}}
    examples.append(make_example(state, history, tower_positions, dec))
    history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
    history.append({"role": "user", "content": f"{events}개"})
    state["target_events"] = events
    state["phase"] = "idle"

    dec = {"tool": "none", "update_state": {"target_events": events, "phase": "idle"}}
    examples.append(make_example(state, history, tower_positions, dec))
    history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})

    # --- STEP 1 ---
    run_number = 100
    for i, tower in enumerate(TOWER_ORDER):
        state["current_tower_idx"] = i
        pos = tower_positions[tower]

        dec = {"message": MESSAGE_MOVE_REQ.format(tower=tower, x=pos['x'], y=pos['y']),
               "update_state": {"current_tower": tower}}
        examples.append(make_example(state, history, tower_positions, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})

        dec = {"tool": "daq_run_tool", "params": {
            "events": events, "config": "setup",
            "pos_h": pos['x'], "pos_v": pos['y'],
            "pos_rot": 0.0, "pos_tilt": 0.0, "beam_energy": energy
        }}
        examples.append(make_example(state, history, tower_positions, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        state['tower_status'][tower]['collected_events'] = events
        state['tower_status'][tower]['runs'].append(run_number)

        dec = {"message": MESSAGE_PLOT_CONFIRM}
        examples.append(make_example(state, history, tower_positions, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})

        dec = {"tool": "none", "update_state": {
            "tower_status": {tower: {"completed": True}}
        }}
        examples.append(make_example(state, history, tower_positions, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        state['tower_status'][tower]['completed'] = True
        # idx는 tower_status 완료 수 기반으로 자동 산출 (agent와 동일)
        state['current_tower_idx'] = sum(1 for s in state['tower_status'].values() if s['completed'])
        run_number += 1

    # --- STEP 2 ---
    dec = {"message": MESSAGE_COMPLETE}
    examples.append(make_example(state, history, tower_positions, dec))
    return examples


def generate_workflow_from_mid(energy: float, events: int, start_idx: int) -> List[Dict[str, Any]]:
    """후반 타워부터 시작하는 partial 워크플로우 — 히스토리 창 밖 앞부분을 건너뛰어
    모델이 후반 타워(T7, T8 등)에서도 올바른 결정을 내리도록 학습"""
    examples = []
    tower_positions = {t: {'x': round(random.uniform(500, 5000), 1),
                           'y': round(random.uniform(500, 5000), 1)} for t in TOWER_ORDER}

    # start_idx 이전 타워는 이미 완료된 상태로 초기화
    state = {
        "phase": "idle",
        "beam_energy": energy,
        "target_events": events,
        "current_tower_idx": start_idx,
        "tower_status": {
            t: {
                "collected_events": events if j < start_idx else 0,
                "runs": [1000 + j] if j < start_idx else [],
                "completed": j < start_idx,
            }
            for j, t in enumerate(TOWER_ORDER)
        }
    }

    # 이전 타워들 완료 이력을 히스토리에 압축해 넣어 10턴 창 내에 배치
    history = []
    for j in range(max(0, start_idx - 2), start_idx):
        prev_tower = TOWER_ORDER[j]
        prev_pos = tower_positions[prev_tower]
        prev_run = 1000 + j
        history.append({"role": "assistant", "content": json.dumps(
            {"message": MESSAGE_MOVE_REQ.format(tower=prev_tower, x=prev_pos['x'], y=prev_pos['y']),
             "update_state": {"current_tower": prev_tower}}, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})
        history.append({"role": "assistant", "content": json.dumps(
            {"tool": "daq_run_tool", "params": {"events": events, "config": "setup",
             "pos_h": prev_pos['x'], "pos_v": prev_pos['y'],
             "pos_rot": 0.0, "pos_tilt": 0.0, "beam_energy": energy}}, ensure_ascii=False)})
        history.append({"role": "assistant", "content": json.dumps(
            {"message": MESSAGE_PLOT_CONFIRM}, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})
        history.append({"role": "assistant", "content": json.dumps(
            {"tool": "none", "update_state": {
                "tower_status": {prev_tower: {"completed": True}}}}, ensure_ascii=False)})

    run_number = 1000 + start_idx
    for i in range(start_idx, len(TOWER_ORDER)):
        tower = TOWER_ORDER[i]
        state["current_tower_idx"] = i
        pos = tower_positions[tower]

        dec = {"message": MESSAGE_MOVE_REQ.format(tower=tower, x=pos['x'], y=pos['y']),
               "update_state": {"current_tower": tower}}
        examples.append(make_example(state, history, tower_positions, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})

        dec = {"tool": "daq_run_tool", "params": {
            "events": events, "config": "setup",
            "pos_h": pos['x'], "pos_v": pos['y'],
            "pos_rot": 0.0, "pos_tilt": 0.0, "beam_energy": energy
        }}
        examples.append(make_example(state, history, tower_positions, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        state['tower_status'][tower]['collected_events'] = events
        state['tower_status'][tower]['runs'].append(run_number)

        dec = {"message": MESSAGE_PLOT_CONFIRM}
        examples.append(make_example(state, history, tower_positions, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})

        dec = {"tool": "none", "update_state": {
            "tower_status": {tower: {"completed": True}}
        }}
        examples.append(make_example(state, history, tower_positions, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        state['tower_status'][tower]['completed'] = True
        state['current_tower_idx'] = sum(1 for s in state['tower_status'].values() if s['completed'])
        run_number += 1

    dec = {"message": MESSAGE_COMPLETE}
    examples.append(make_example(state, history, tower_positions, dec))
    return examples


def main():
    output_file = Path(__file__).parent / "data" / "calib_scan_data.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_ex = []
    for energy in [2, 10, 20, 50, 100, 120]:
        for events in [500, 1000, 3000]:
            all_ex.extend(generate_workflow_normal(energy, events))
    # 후반 타워에서 시작하는 partial 워크플로우 (T4, T7, T8 위주)
    for energy in [2, 10, 50]:
        for events in [500, 1000]:
            for start_idx in [3, 6, 7]:   # T6(3), T7(6), T8(7)
                all_ex.extend(generate_workflow_from_mid(energy, events, start_idx))
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in all_ex:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    lengths = [sum(len(m["content"]) for m in ex["messages"]) for ex in all_ex]
    max_chars = max(lengths)
    avg_chars = sum(lengths) / len(lengths)
    print(f"✅ Generated {len(all_ex)} samples → {output_file}")
    print(f"   char len  max={max_chars:,}  avg={avg_chars:,.0f}  (≈token max={max_chars//2:,}  avg={avg_chars//2:,.0f})")


if __name__ == "__main__":
    main()

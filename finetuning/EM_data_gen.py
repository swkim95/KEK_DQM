#!/usr/bin/env python3
"""
Training data generator for Energy Scan Agent
- build_full_context, _build_state_context, _get_step_hint 포맷이
  energy_scan_agent.py와 완전히 동일하도록 유지
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

MESSAGE_T5_MOVE = "T5 타워 중심으로 이동해주세요 (x:{x}, y:{y})."
MESSAGE_ENERGY_SET = "빔 에너지를 {energy} GeV로 설정해주세요."
MESSAGE_PLOT_CONFIRM = "데이터 수집 및 Plot 생성이 완료되었습니다. 결과를 확인해주세요."
MESSAGE_COMPLETE = "모든 에너지 스캔이 완료되었습니다."

SYSTEM_PROMPT = """You are Energy Scan Agent for test beam experiments.

Follow these steps EXACTLY:

=== STEP 0: Parse Energy Config ===
Tool: "none"
Message: (none)
Update State: {"energy_config": {...}, "scan_order": [list], "phase": "idle"}
Note: scan_order MUST match energy_config keys (integers)
CRITICAL: If user says "모두", "각각", or "씩" with one number (e.g., "모두 500개"), apply that number to ALL energies.

=== STEP 1: Request T5 Movement ===
CRITICAL RULE: After STEP 0, when phase is "idle" and energy_config is NOT empty, you MUST output T5 movement message.
DO NOT repeat STEP 0. DO NOT skip to phase "scanning". 
You MUST output: {"message": "T5 타워 중심으로 이동해주세요 (x:{x}, y:{y})."}
Note: replace {x}, {y} with the actual coordinates provided in the state.

After user says "완료":
CRITICAL: When you see "완료" in conversation history, you MUST NOT repeat the same message.
You MUST proceed to next step immediately by outputting:
{"tool": "none", "update_state": {"phase": "scanning"}}
DO NOT output the same message again. DO NOT ask for T5 movement again.

Then go to STEP 2.

=== STEP 2: For Each Energy in scan_order (REPEAT for ALL energies) ===
Repeat steps 2a-2c for each energy in scan_order until all energies are completed.

2a. Request Energy Setting
Output: {"message": "빔 에너지를 {energy} GeV로 설정해주세요."} (replace {energy} with number, e.g., "빔 에너지를 10 GeV로 설정해주세요.")

After user says "완료":
CRITICAL: When you see "완료" in conversation history, proceed to next step immediately.
2b. Execute DAQ
Tool: "daq_run_tool"
Params: {
    "events": <target_events from energy_config>, 
    "config": "setup",
    "pos_h": <x_from_state>,
    "pos_v": <y_from_state>,
    "pos_rot": 1.5,
    "pos_tilt": 1.0,
    "beam_energy": <energy>
}
(Plot is auto-rendered by DQM live during DAQ — never call any plot tool.)

2c. Request Plot Confirmation
Output: {"message": "데이터 수집 및 Plot 생성이 완료되었습니다. 결과를 확인해주세요."}

After user says "완료":
CRITICAL: When you see "완료" in conversation history, you MUST mark current energy as completed.
Output: {"tool": "none", "update_state": {"energy_config": {"{energy}": {"completed": true}}}}
Then proceed to next energy or STEP 3.

=== STEP 3: Completion ===
When ALL energies in energy_config have "completed": true, you MUST output:
{"message": "모든 에너지 스캔이 완료되었습니다."}

After this message, if user provides NEW energy settings, you MUST reset and go back to STEP 0.
If user says "종료", the session ends.

=== CRITICAL RULES ===
1. Follow steps STRICTLY in order. Do NOT skip or reorder steps.
2. Use EXACT messages above. DO NOT change or paraphrase.
3. Output JSON format (CHOOSE ONE, NEVER BOTH):
   - {"tool": "...", "params": {...}, "update_state": {...}}  (for tool execution)
   - {"message": "...", "update_state": {...}}  (for user message)
   - {"tool": "none", "update_state": {...}}  (for internal state update only)
   CRITICAL: NEVER output both "tool" and "message" in the same JSON. NEVER put "message" inside "update_state".
4. Use energy_config[energy].target_events for DAQ events
5. STEP TRANSITION RULES (MOST CRITICAL):
   - After STEP 0 (energy_config parsed, phase="idle"): You MUST go to STEP 1 (T5 movement message). DO NOT repeat STEP 0.
   - After STEP 1 (T5 movement message sent): Wait for user "완료", then go to STEP 2.
   - NEVER skip STEP 1. NEVER output STEP 0 decision twice in a row.
6. MOST IMPORTANT: When you see "완료" in conversation history, you MUST process it IMMEDIATELY:
   - T5 movement "완료" → Output {"tool": "none", "update_state": {"phase": "scanning"}} and proceed to STEP 2
   - Energy setting "완료" → Execute DAQ immediately ({"tool": "daq_run_tool", ...})
   - Plot confirmation "완료" → Update energy status to "completed": true.
7. AFTER COMPLETION: If all energies are done and you have sent the completion message, you are ready for a new task or exit.
"""


def _build_state_context(state):
    lines = []
    lines.append(f"Phase: {state.get('phase', 'None')}")
    lines.append(f"Tower: {state.get('tower', 'T5')}")
    t5_x = state.get('t5_x', 1234.0)
    t5_y = state.get('t5_y', 1234.0)
    lines.append(f"T5 Position: x={t5_x:.1f}, y={t5_y:.1f}, rot=1.5, tilt=1.0")
    if state.get('position'):
        lines.append(f"Position: {state['position']}")
    lines.append("")
    if state.get('scan_order'):
        lines.append(f"Scan Order: {state['scan_order']}")
        lines.append(f"Current Energy: {state.get('current_energy')} GeV (index: {state.get('current_energy_idx', 0)})")
        lines.append("")
        lines.append("Energy Progress:")
        for energy in state['scan_order']:
            if energy is None:
                continue
            config = state['energy_config'].get(energy, {})
            collected = config.get('collected_events', 0)
            target = config.get('target_events', 0)
            runs = config.get('runs', [])
            completed = config.get('completed', False)
            if completed:
                status = "✅"
            elif collected > 0:
                status = f"⏳ {collected}/{target}"
            else:
                status = "⏸️  Not started"
            run_info = f"Runs: {runs}" if runs else ""
            if energy == state.get('current_energy'):
                lines.append(f"  {energy} GeV: {status} {run_info} ← CURRENT (target: {target} events)")
            else:
                lines.append(f"  {energy} GeV: {status} {run_info}")
    return "\n".join(lines)


def _build_history_context(history):
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


def _get_step_hint(state, history):
    """agent의 _get_step_hint()와 동일 - 상태 요약만, AI가 스스로 결정"""
    phase = state.get("phase", "config")
    current_energy = state.get("current_energy")
    scan_order = state.get("scan_order", [])
    idx = scan_order.index(current_energy) + 1 if current_energy in scan_order else 0
    total = len(scan_order)
    return f"Phase: {phase} | Energy: {current_energy} GeV ({idx}/{total})"


def build_full_context(state, history, current_input=None):
    if current_input is None and history and history[-1]["role"] == "user":
        current_input = history[-1]["content"]
        temp_history = history[:-1]
    else:
        temp_history = history

    parts = []
    parts.append("=== Current State ===")
    parts.append(_build_state_context(state))
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


def make_example(state, history, decision, current_input=None):
    ctx = build_full_context(state, history, current_input)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ctx},
            {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)}
        ]
    }


def generate_workflow_normal(energy_list, events_list, user_input):
    examples = []
    history = []
    t5_x = round(random.uniform(1000.0, 2000.0), 1)
    t5_y = round(random.uniform(1000.0, 2000.0), 1)
    state = {
        "phase": "config", "tower": "T5", "t5_x": t5_x, "t5_y": t5_y,
        "position": {"x": 0.2, "y": -0.3},
        "energy_config": {}, "scan_order": [],
        "current_energy": None, "current_energy_idx": 0,
        "plot_method": "PeakADC", "plot_max_event": None
    }

    energy_config_dict = {
        e: {"target_events": ev, "collected_events": 0, "runs": [], "completed": False, "completed_at": None}
        for e, ev in zip(energy_list, events_list)
    }
    dec = {"tool": "none", "update_state": {"energy_config": energy_config_dict, "scan_order": energy_list, "phase": "idle"}}
    examples.append(make_example(state, history, dec, user_input))
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
    state.update(dec["update_state"])

    dec = {"message": MESSAGE_T5_MOVE.format(x=t5_x, y=t5_y)}
    examples.append(make_example(state, history, dec))
    history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
    history.append({"role": "user", "content": "완료"})

    dec = {"tool": "none", "update_state": {"phase": "scanning"}}
    examples.append(make_example(state, history, dec))
    history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
    state["phase"] = "scanning"

    run_number = 100
    for i, energy in enumerate(energy_list):
        events = events_list[i]
        state["current_energy"] = energy
        state["current_energy_idx"] = i

        dec = {"message": MESSAGE_ENERGY_SET.format(energy=energy)}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})

        dec = {"tool": "daq_run_tool", "params": {
            "events": events, "config": "setup",
            "pos_h": t5_x, "pos_v": t5_y,
            "pos_rot": 1.5, "pos_tilt": 1.0, "beam_energy": energy
        }}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        state["energy_config"][energy]["collected_events"] = events
        state["energy_config"][energy]["runs"].append(run_number)

        dec = {"message": MESSAGE_PLOT_CONFIRM}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})

        dec = {"tool": "none", "update_state": {"energy_config": {str(energy): {"completed": True}}}}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        state["energy_config"][energy]["completed"] = True
        run_number += 1

    dec = {"message": MESSAGE_COMPLETE}
    examples.append(make_example(state, history, dec))
    return examples


def generate_workflow_from_mid(energy_list: list, events_list: list, start_idx: int) -> list:
    """후반 에너지부터 시작하는 partial 워크플로우.
    모델이 3~4번째 에너지에서도 올바른 결정을 내리도록 학습.
    calib_data_gen의 generate_workflow_from_mid과 동일한 원리."""
    examples = []
    t5_x = round(random.uniform(1000.0, 2000.0), 1)
    t5_y = round(random.uniform(1000.0, 2000.0), 1)

    energy_config_dict = {
        e: {
            "target_events": ev,
            "collected_events": ev if j < start_idx else 0,
            "runs": [1000 + j] if j < start_idx else [],
            "completed": j < start_idx,
            "completed_at": "10:00:00" if j < start_idx else None,
        }
        for j, (e, ev) in enumerate(zip(energy_list, events_list))
    }

    state = {
        "phase": "scanning", "tower": "T5", "t5_x": t5_x, "t5_y": t5_y,
        "position": None,
        "energy_config": energy_config_dict,
        "scan_order": energy_list,
        "current_energy": energy_list[start_idx] if start_idx < len(energy_list) else None,
        "current_energy_idx": start_idx,
        "plot_method": "PeakADC", "plot_max_event": None,
    }

    # 앞선 에너지의 완료 이력을 history에 압축 (최근 2개만)
    history = []
    for j in range(max(0, start_idx - 2), start_idx):
        prev_e = energy_list[j]
        prev_ev = events_list[j]
        prev_run = 1000 + j
        history.append({"role": "assistant", "content": json.dumps(
            {"message": MESSAGE_ENERGY_SET.format(energy=prev_e)}, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})
        history.append({"role": "assistant", "content": json.dumps(
            {"tool": "daq_run_tool", "params": {
                "events": prev_ev, "config": "setup",
                "pos_h": t5_x, "pos_v": t5_y,
                "pos_rot": 1.5, "pos_tilt": 1.0, "beam_energy": prev_e,
            }}, ensure_ascii=False)})
        history.append({"role": "assistant", "content": json.dumps(
            {"message": MESSAGE_PLOT_CONFIRM}, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})
        history.append({"role": "assistant", "content": json.dumps(
            {"tool": "none", "update_state": {
                "energy_config": {str(prev_e): {"completed": True}}}}, ensure_ascii=False)})

    run_number = 1000 + start_idx
    for i in range(start_idx, len(energy_list)):
        energy = energy_list[i]
        events = events_list[i]
        state["current_energy"] = energy
        state["current_energy_idx"] = i

        dec = {"message": MESSAGE_ENERGY_SET.format(energy=energy)}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})

        dec = {"tool": "daq_run_tool", "params": {
            "events": events, "config": "setup",
            "pos_h": t5_x, "pos_v": t5_y,
            "pos_rot": 1.5, "pos_tilt": 1.0, "beam_energy": energy,
        }}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        state["energy_config"][energy]["collected_events"] = events
        state["energy_config"][energy]["runs"].append(run_number)

        dec = {"message": MESSAGE_PLOT_CONFIRM}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})

        dec = {"tool": "none", "update_state": {"energy_config": {str(energy): {"completed": True}}}}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        state["energy_config"][energy]["completed"] = True
        run_number += 1

    dec = {"message": MESSAGE_COMPLETE}
    examples.append(make_example(state, history, dec))
    return examples


def main():
    output_file = Path(__file__).parent / "data" / "EM_scan_data.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_ex = []

    # ── 전체 워크플로우 (STEP 0 → STEP 3) ──────────────────────────────────
    test_cases = [
        # ── 소에너지 1~5 GeV (모델이 "1→10 오인" 방지용) ──
        ([1, 2, 3],       [100, 200, 400],        "1GeV 100개, 2GeV 200개, 3GeV 400개"),
        ([1, 2, 3, 4],    [100, 100, 100, 100],   "1,2,3,4GeV 각각 100개씩"),
        ([1, 3, 5],       [500, 500, 500],         "1,3,5GeV 모두 500개"),
        ([1, 2],          [200, 200],              "1GeV 200개 2GeV 200개"),
        ([1, 2, 3, 4],    [1000, 1000, 1000, 1000],"1,2,3,4GeV 모두 1000개"),
        ([1, 5],          [500, 500],              "1GeV 500개, 5GeV 500개"),
        ([1, 2, 3],       [10000, 10000, 10000],   "1,2,3GeV 각각 1만개"),
        ([2, 4],          [300, 300],              "2,4GeV 각각 300개"),
        ([3, 5],          [1000, 1000],            "3GeV 1000개 5GeV 1000개"),
        ([1, 5, 10, 20],  [500, 500, 500, 500],   "1,5,10,20GeV 모두 500개"),
        ([1, 5, 10, 20],  [1000, 2000, 3000, 5000],"1GeV 1000개, 5GeV 2000개, 10GeV 3000개, 20GeV 5000개"),
        # ── 중간 에너지 5~30 GeV ──
        ([5, 10],         [300, 300],              "5,10GeV 모두 300개"),
        ([10, 20, 30],    [500, 500, 500],         "10,20,30GeV 각각 500개씩"),
        ([10, 20, 30],    [1000, 1000, 1000],      "10,20,30GeV 모두 1000개"),
        ([10, 20, 30],    [500, 500, 500],         "10,20,30GeV 500개씩"),
        ([10, 20, 30],    [1000, 1000, 1000],      "에너지 10,20,30 모두 1000개"),
        ([10, 20, 30],    [100, 100, 100],         "에너지 10,20,30GeV에 대해 각각 100개씩"),
        ([5, 10, 20, 30], [500, 500, 1000, 1000],  "5GeV 500개, 10GeV 500개, 20GeV 1000개, 30GeV 1000개"),
        # ── 소+중 혼합 ──
        ([2, 5, 10],      [100, 200, 300],         "2GeV 100개, 5GeV 200개, 10GeV 300개"),
        ([2, 5, 10, 20],  [500, 500, 500, 500],   "에너지는 2,5,10,20이고 모두 500개"),
        ([2, 5, 10, 20],  [10, 20, 30, 40],       "2GeV 10개, 5GeV 20개, 10GeV 30개, 20GeV 40개"),
        ([2, 5, 10, 20],  [5000, 5000, 5000, 5000],"2,5,10,20GeV 각각 5000개"),
        # ── 고에너지 40~120 GeV ──
        ([20, 40, 60],    [1000, 1000, 1500],      "20GeV 1000개, 40GeV 1000개, 60GeV 1500개"),
        ([40, 60, 80, 100],[1000, 1000, 1000, 1000],"40,60,80,100GeV 모두 1000개"),
        ([50, 100],       [2000, 2000],            "50GeV 2000개 100GeV 2000개"),
        ([20, 50, 100, 120],[500, 1000, 2000, 3000],"20GeV 500개, 50GeV 1000개, 100GeV 2000개, 120GeV 3000개"),
        # ── 단일 에너지 ──
        ([10],            [1000],                  "10GeV 1000개"),
        ([1],             [500],                   "1GeV 500개"),
        ([5],             [2000],                  "5GeV 2000개"),
    ]
    for energy_list, events_list, user_input in test_cases:
        all_ex.extend(generate_workflow_normal(energy_list, events_list, user_input))

    # ── 후반 에너지부터 시작하는 partial 워크플로우 ─────────────────────────
    mid_cases = [
        # 소에너지 리스트
        ([1, 2, 3, 4],    [100, 100, 100, 100],   2),
        ([1, 2, 3, 4],    [100, 100, 100, 100],   3),
        ([1, 2, 3],       [1000, 1000, 1000],      1),
        ([1, 2, 3],       [1000, 1000, 1000],      2),
        ([1, 5, 10, 20],  [500, 500, 500, 500],   2),
        ([1, 5, 10, 20],  [500, 500, 500, 500],   3),
        # 중간 에너지 리스트
        ([10, 20, 30],    [500, 500, 500],         1),
        ([10, 20, 30],    [500, 500, 500],         2),
        ([5, 10, 20, 30], [500, 500, 1000, 1000],  2),
        ([5, 10, 20, 30], [500, 500, 1000, 1000],  3),
        # 혼합 리스트
        ([2, 5, 10, 20],  [500, 500, 500, 500],   2),
        ([2, 5, 10, 20],  [500, 500, 500, 500],   3),
        ([20, 40, 60],    [1000, 1000, 1500],      1),
        ([20, 40, 60],    [1000, 1000, 1500],      2),
        ([40, 60, 80, 100],[1000, 1000, 1000, 1000],2),
        ([40, 60, 80, 100],[1000, 1000, 1000, 1000],3),
    ]
    for energy_list, events_list, start_idx in mid_cases:
        all_ex.extend(generate_workflow_from_mid(energy_list, events_list, start_idx))

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

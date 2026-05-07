#!/usr/bin/env python3
"""
Training data generator for HV Equalization Agent
- build_full_context, _build_state_context, _get_step_hint 포맷이
  hv_equalization_agent.py와 완전히 동일하도록 유지
- 단일 타워 워크플로우 (컨트롤러가 타워 반복, agent는 1개 타워만 처리)
"""

import json
import random
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

TOWER_ORDER = ["T1", "T2", "T3", "T6", "T5", "T4", "T7", "T8", "T9"]

# 메시지 템플릿 (agent와 동일)
MESSAGE_MOVE_REQ    = "{tower} 타워 중심으로 이동해주세요 (x:{x:.1f}, y:{y:.1f})."
MESSAGE_HV_CONFIRM  = "HV 전압이 변경되었습니다. 결과를 확인하고 '완료'를 입력하면 다음 DAQ를 시작합니다."
MESSAGE_APPROVE_BOTH = "분석 결과, 현재 ADC: {tower}C={adc_c:.1f}, {tower}S={adc_s:.1f} (목표: {target}). HV 변경 제안: {tower}C {hv_c_old}V→{hv_c_new}V, {tower}S {hv_s_old}V→{hv_s_new}V. 적용하시겠습니까?"
MESSAGE_APPROVE_C   = "분석 결과, 현재 ADC: {tower}C={adc_c:.1f} (목표: {target}). HV 변경 제안: {tower}C {hv_c_old}V→{hv_c_new}V. ({tower}S 완료) 적용하시겠습니까?"
MESSAGE_APPROVE_S   = "분석 결과, 현재 ADC: {tower}S={adc_s:.1f} (목표: {target}). HV 변경 제안: {tower}S {hv_s_old}V→{hv_s_new}V. ({tower}C 완료) 적용하시겠습니까?"


def _make_system_prompt(tower: str, x: float, y: float) -> str:
    """agent의 _get_system_prompt()와 동일"""
    t = tower
    return f"""You are HV Equalization Agent for tower {t} (x:{x:.1f}, y:{y:.1f}).
Your task: adjust HV for {t}C and {t}S channels to reach the target peakADC value.

Follow these steps EXACTLY:

=== Workflow for {t} ===
1a. Request move:
  {{"message": "{t} 타워 중심으로 이동해주세요 (x:{x:.1f}, y:{y:.1f})."}}

After user says "완료":
1b. Check HV status:
  {{"tool": "hv_execute_tool", "params": {{"command": "status", "channels": ["{t}C", "{t}S"]}}}}

[INNER LOOP — repeat 1c→1g until CONVERGED]
1c. Execute DAQ:
  {{"tool": "daq_run_tool", "params": {{"events": <events>, "pos_h": <x>, "pos_v": <y>, "beam_energy": <energy>}}}}
  (Plot is auto-rendered by DQM live during DAQ — never call any plot tool.)

1d. Suggest HV:
  {{"tool": "hv_equalization_suggest", "params": {{"run_number": <run>, "tower": "{t}"}}}}

1e. Ask approval (only NOT-done channels):
  Both not done: {{"message": "분석 결과, 현재 ADC: {t}C=<adc_c>, {t}S=<adc_s> (목표: <target>). HV 변경 제안: {t}C <old_c>V→<new_c>V, {t}S <old_s>V→<new_s>V. 적용하시겠습니까?", "update_state": {{"phase": "approving"}}}}
  Only C not done: {{"message": "분석 결과, 현재 ADC: {t}C=<adc_c> (목표: <target>). HV 변경 제안: {t}C <old_c>V→<new_c>V. ({t}S 완료) 적용하시겠습니까?", "update_state": {{"phase": "approving"}}}}
  Only S not done: {{"message": "분석 결과, 현재 ADC: {t}S=<adc_s> (목표: <target>). HV 변경 제안: {t}S <old_s>V→<new_s>V. ({t}C 완료) 적용하시겠습니까?", "update_state": {{"phase": "approving"}}}}
  CRITICAL: Use EXACT values from state — NEVER fabricate numbers.

After user says "완료":
1f. Apply voltage (only NOT-done channels):
  {{"tool": "hv_execute_tool", "params": {{"command": "voltage", "channel_values": {{"{t}C": <new_c>, "{t}S": <new_s>}}}}, "update_state": {{"phase": "equalizing"}}}}
  NEVER include a done channel in channel_values.

1g. Confirmation:
  {{"message": "HV 전압이 변경되었습니다. 결과를 확인하고 '완료'를 입력하면 다음 DAQ를 시작합니다."}}

After user says "완료":
  - State shows NOT CONVERGED → back to step 1c
  - State shows CONVERGED (C=True, S=True) → proceed to step 1h

1h. Done:
  {{"tool": "hv_equalization_done_channel", "params": {{"channels": "all"}}}}

=== CRITICAL RULES ===
1. Follow steps STRICTLY in order. Do NOT skip Step 1e (Approval).
2. Output JSON ONLY. No natural language.
3. NEVER include a done channel in channel_values.
4. ALWAYS use EXACT numbers from state — never invent values.
5. When CONVERGED (state C=True, S=True), call hv_equalization_done_channel IMMEDIATELY.
"""


def _build_state_context(state: Dict) -> str:
    """agent의 _build_state_context()와 동일"""
    lines = []
    tower = state["current_tower"]
    pos_x = state.get("tower_pos", {}).get("x", 0.0)
    pos_y = state.get("tower_pos", {}).get("y", 0.0)
    adc_known = state.get("last_adc_c") is not None
    suggest_pending = state.get("last_suggested_hv_c") is not None
    phase = state.get("phase", "idle")

    if state.get("last_hv_c") is None:
        lines.append(f"*** REQUIRED NEXT: move request (step 1a) — ask user to move to {tower} ***")
        lines.append("")

    if adc_known:
        done_c = state.get("channel_done_c", False)
        done_s = state.get("channel_done_s", False)
        adc_c = state["last_adc_c"]
        adc_s = state["last_adc_s"]
        target = state.get("target_adc_c")
        if done_c and done_s:
            lines.append(f"*** CONVERGENCE: C=True, S=True — CALL hv_equalization_done_channel NOW ***")
        elif suggest_pending and phase == "approving":
            lines.append(f"*** CONVERGENCE: C={done_c}, S={done_s} | ADC: C={adc_c:.1f}, S={adc_s:.1f} | Target: {target} ***")
            if done_c:
                lines.append(f"*** C=DONE: NEVER include {tower}C in channel_values ***")
            if done_s:
                lines.append(f"*** S=DONE: NEVER include {tower}S in channel_values ***")
            lines.append(f"*** REQUIRED NEXT: hv_execute_tool voltage (step 1f) ***")
        elif suggest_pending:
            lines.append(f"*** CONVERGENCE: C={done_c}, S={done_s} | ADC: C={adc_c:.1f}, S={adc_s:.1f} | Target: {target} ***")
            if done_c:
                lines.append(f"*** C=DONE: NEVER include {tower}C in channel_values ***")
            if done_s:
                lines.append(f"*** S=DONE: NEVER include {tower}S in channel_values ***")
            lines.append(f"*** REQUIRED NEXT: approval message (step 1e) ***")
        else:
            lines.append(f"*** CONVERGENCE: C={done_c}, S={done_s} | ADC: C={adc_c:.1f}, S={adc_s:.1f} | Target: {target} ***")
            if done_c:
                lines.append(f"*** C=DONE: NEVER include {tower}C in channel_values ***")
            if done_s:
                lines.append(f"*** S=DONE: NEVER include {tower}S in channel_values ***")
            lines.append(f"*** REQUIRED NEXT: daq_run_tool (step 1c) ***")
        lines.append("")

    lines.append(f"Phase: {phase}")
    lines.append(f"Tower: {tower} (x:{pos_x:.1f}, y:{pos_y:.1f})")
    lines.append(f"Beam Energy: {state.get('beam_energy')} GeV")
    lines.append(f"Target Events: {state.get('target_events')}")
    lines.append(f"Target ADC: {state.get('target_adc_c')}")
    lines.append(f"Last HV: C={state.get('last_hv_c')}V, S={state.get('last_hv_s')}V")
    if state.get("last_suggested_hv_c") is not None:
        dc = state.get("channel_done_c", False)
        ds = state.get("channel_done_s", False)
        c_str = f"C={state['last_suggested_hv_c']}V" + (" [DONE-skip]" if dc else "")
        s_str = f"S={state['last_suggested_hv_s']}V" + (" [DONE-skip]" if ds else "")
        lines.append(f"Suggested HV: {c_str}, {s_str}  <- use EXACT values in approval message")
    if state.get("last_run_number"):
        lines.append(f"Last Run Number: {state['last_run_number']}")
    lines.append(f"Iterations: {state.get('iterations', 0)}")
    return "\n".join(lines)


def _build_history_context(history: List[Dict]) -> str:
    """agent의 build_full_context() 내 history 처리와 동일"""
    if not history:
        return "(No conversation yet)"
    lines = []
    for msg in history[-10:]:
        role = "User" if msg["role"] == "user" else "Agent"
        content = msg["content"]
        if content.strip() == "완료":
            lines.append(f"{role}: 완료 [IMPORTANT: User confirmed completion]")
        else:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _get_step_hint(state: Dict) -> str:
    """agent의 _get_step_hint()와 동일"""
    tower = state["current_tower"]
    adc_known = state.get("last_adc_c") is not None
    suggest_pending = state.get("last_suggested_hv_c") is not None
    done_c = state.get("channel_done_c", False)
    done_s = state.get("channel_done_s", False)
    phase = state.get("phase", "idle")
    base = f"Phase: {phase} | Tower: {tower}"

    if state.get("last_hv_c") is None:
        return f"{base} | REQUIRED NEXT: move request (step 1a)"
    elif adc_known and done_c and done_s:
        return f"{base} | CONVERGED → call hv_equalization_done_channel (step 1h)"
    elif adc_known and suggest_pending and phase == "approving":
        return f"{base} | REQUIRED NEXT: hv_execute_tool voltage (step 1f — user already confirmed)"
    elif adc_known and suggest_pending:
        return f"{base} | REQUIRED NEXT: approval message (step 1e)"
    elif adc_known:
        return f"{base} | REQUIRED NEXT: daq_run_tool (step 1c)"
    else:
        return f"{base} | REQUIRED NEXT: daq_run_tool (step 1c — first DAQ)"


def build_full_context(state: Dict, history: List[Dict], current_input: Optional[str] = None) -> str:
    """agent의 build_full_context()와 동일한 구조"""
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
    parts.append(_get_step_hint(state))
    parts.append("")
    parts.append("Output JSON with tool name and parameters.")
    return "\n".join(parts)


def make_example(state: Dict, history: List[Dict], decision: Dict,
                 current_input: Optional[str] = None) -> Dict:
    tower = state["current_tower"]
    pos_x = state.get("tower_pos", {}).get("x", 0.0)
    pos_y = state.get("tower_pos", {}).get("y", 0.0)
    system_prompt = _make_system_prompt(tower, pos_x, pos_y)
    ctx = build_full_context(state, history, current_input)
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": ctx},
            {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)},
        ]
    }


# ─────────────────────────────────────────────────────────────
# Convergence patterns: list of (done_c, done_s) per iteration
# ─────────────────────────────────────────────────────────────
# 각 패턴은 [(done_c_iter1, done_s_iter1), (done_c_iter2, done_s_iter2), ...]
# 마지막 항목은 항상 (True, True)
CONVERGENCE_PATTERNS = [
    [(True,  True)],                                           # 1회: 즉시 수렴
    [(False, False), (True,  True)],                           # 2회: 둘 다 수렴
    [(True,  False), (True,  True)],                           # 2회: C 먼저
    [(False, True),  (True,  True)],                           # 2회: S 먼저
    [(False, False), (False, False), (True,  True)],           # 3회
    [(False, False), (True,  False), (True,  True)],           # 3회: C가 2회에 수렴
    [(False, False), (False, True),  (True,  True)],           # 3회: S가 2회에 수렴
    [(False, False), (False, False), (False, False), (True, True)],  # 4회 느린 수렴
    [(True,  False), (True,  False), (True,  True)],           # C 먼저, S 느림
    # HV 감소 시나리오 (ADC가 target 초과 → HV를 낮춰야)
    [(False, False), (True,  True)],   # 감소 패턴도 동일 구조 (HV delta로 구분)
]


def _hv_delta(adc_frac: float, going_down: bool) -> int:
    """ADC 비율에 따른 HV 변화량 결정 (실제 방향 반영)"""
    if going_down:
        # ADC가 target 초과 → HV 감소
        return random.randint(-15, -6)
    else:
        # ADC가 target 미달 → HV 증가
        return random.randint(6, 18)


def generate_workflow(tower: str, x: float, y: float,
                      energy: float, events: int, target_adc: float,
                      pattern: List, going_down: bool = False,
                      start_iter: int = 0) -> List[Dict]:
    """단일 타워 워크플로우 훈련 데이터 생성.
    start_iter > 0이면 중간 상태에서 시작 (모델이 후반 반복에서도 학습).
    """
    examples = []
    history = []

    hv_c = random.randint(820, 880)
    hv_s = random.randint(820, 880)
    run_number = random.randint(10000, 19999)

    state = {
        "phase": "idle",
        "beam_energy": energy,
        "target_events": events,
        "target_adc_c": target_adc,
        "target_adc_s": target_adc,
        "current_tower": tower,
        "tower_pos": {"x": x, "y": y},
        "last_hv_c": None,
        "last_hv_s": None,
        "last_suggested_hv_c": None,
        "last_suggested_hv_s": None,
        "last_adc_c": None,
        "last_adc_s": None,
        "channel_done_c": False,
        "channel_done_s": False,
        "last_run_number": None,
        "iterations": 0,
    }

    # start_iter > 0: 이미 완료된 반복을 히스토리에 압축 (창 밖 시뮬레이션)
    if start_iter > 0:
        for idx in range(min(start_iter, len(pattern) - 1)):
            iter_done_c, iter_done_s = pattern[idx]
            adc_frac = 0.80 + 0.06 * idx if not going_down else 1.10 - 0.05 * idx
            adc_c = round(target_adc * adc_frac + random.uniform(-5, 5), 1)
            adc_s = round(target_adc * adc_frac + random.uniform(-5, 5), 1)
            delta_c = _hv_delta(adc_frac, going_down)
            delta_s = _hv_delta(adc_frac, going_down)
            next_hv_c = hv_c + delta_c
            next_hv_s = hv_s + delta_s

            # 이전 반복 히스토리 압축 (최근 2회만)
            if idx >= start_iter - 2:
                history.append({"role": "assistant", "content": json.dumps(
                    {"tool": "daq_run_tool", "params": {"events": events, "pos_h": x, "pos_v": y, "beam_energy": energy}}
                , ensure_ascii=False)})
                history.append({"role": "assistant", "content": json.dumps(
                    {"tool": "hv_equalization_suggest", "params": {"run_number": run_number, "tower": tower}}
                , ensure_ascii=False)})
                history.append({"role": "assistant", "content": json.dumps(
                    {"message": MESSAGE_APPROVE_BOTH.format(tower=tower, adc_c=adc_c, adc_s=adc_s,
                                                             target=int(target_adc), hv_c_old=hv_c, hv_c_new=next_hv_c,
                                                             hv_s_old=hv_s, hv_s_new=next_hv_s),
                     "update_state": {"phase": "approving"}}
                , ensure_ascii=False)})
                history.append({"role": "user", "content": "완료"})
                history.append({"role": "assistant", "content": json.dumps(
                    {"tool": "hv_execute_tool", "params": {"command": "voltage", "channel_values": {f"{tower}C": next_hv_c, f"{tower}S": next_hv_s}},
                     "update_state": {"phase": "equalizing"}}
                , ensure_ascii=False)})
                history.append({"role": "assistant", "content": json.dumps(
                    {"message": MESSAGE_HV_CONFIRM}
                , ensure_ascii=False)})
                history.append({"role": "user", "content": "완료"})

            hv_c = next_hv_c
            hv_s = next_hv_s
            run_number += 1

        # 중간 상태 세팅
        state["last_hv_c"] = hv_c
        state["last_hv_s"] = hv_s
        state["last_run_number"] = run_number - 1
        state["iterations"] = start_iter
        # 바로 1c(DAQ)부터 시작하므로 adc/suggest 초기화 유지

    # ── 1a: Move request ─────────────────────────────────────
    if start_iter == 0:
        dec = {"message": MESSAGE_MOVE_REQ.format(tower=tower, x=x, y=y)}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})

        # ── 1b: Status check ─────────────────────────────────
        dec = {"tool": "hv_execute_tool", "params": {"command": "status", "channels": [f"{tower}C", f"{tower}S"]}}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        # 상태 업데이트: 초기 HV 셋팅
        state["last_hv_c"] = hv_c
        state["last_hv_s"] = hv_s

    # ── Inner loop ───────────────────────────────────────────
    for iter_idx, (done_c, done_s) in enumerate(pattern):
        adc_frac = 0.80 + 0.07 * (iter_idx + start_iter) if not going_down else 1.12 - 0.06 * (iter_idx + start_iter)
        adc_c = round(target_adc * adc_frac + random.uniform(-8, 8), 1)
        adc_s = round(target_adc * adc_frac + random.uniform(-8, 8), 1)

        # 마지막 반복이면 ADC를 target에 수렴시킴
        if done_c and done_s:
            adc_c = round(target_adc * random.uniform(0.990, 1.010), 1)
            adc_s = round(target_adc * random.uniform(0.990, 1.010), 1)
        elif done_c:
            adc_c = round(target_adc * random.uniform(0.990, 1.010), 1)
        elif done_s:
            adc_s = round(target_adc * random.uniform(0.990, 1.010), 1)

        delta_c = 0 if done_c else _hv_delta(adc_frac, going_down)
        delta_s = 0 if done_s else _hv_delta(adc_frac, going_down)
        next_hv_c = hv_c + delta_c
        next_hv_s = hv_s + delta_s

        # ── 1c: DAQ ──────────────────────────────────────────
        dec = {"tool": "daq_run_tool", "params": {
            "events": events, "pos_h": x, "pos_v": y, "beam_energy": energy
        }}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        state["last_run_number"] = run_number
        state["iterations"] = (state.get("iterations") or 0) + 1

        # ── 1d: Suggest ──────────────────────────────────────
        dec = {"tool": "hv_equalization_suggest", "params": {"run_number": run_number, "tower": tower}}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        # suggest 결과로 state 업데이트
        state["last_adc_c"] = adc_c
        state["last_adc_s"] = adc_s
        state["last_suggested_hv_c"] = next_hv_c
        state["last_suggested_hv_s"] = next_hv_s
        state["channel_done_c"] = done_c
        state["channel_done_s"] = done_s

        if done_c and done_s:
            # ── 1h: Done channel ─────────────────────────────
            dec = {"tool": "hv_equalization_done_channel", "params": {"channels": "all"}}
            examples.append(make_example(state, history, dec))
            history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
            break

        # ── 1e: Approval message ─────────────────────────────
        if not done_c and not done_s:
            msg = MESSAGE_APPROVE_BOTH.format(tower=tower, adc_c=adc_c, adc_s=adc_s,
                                              target=int(target_adc),
                                              hv_c_old=hv_c, hv_c_new=next_hv_c,
                                              hv_s_old=hv_s, hv_s_new=next_hv_s)
        elif done_c:
            msg = MESSAGE_APPROVE_S.format(tower=tower, adc_s=adc_s,
                                           target=int(target_adc),
                                           hv_s_old=hv_s, hv_s_new=next_hv_s)
        else:
            msg = MESSAGE_APPROVE_C.format(tower=tower, adc_c=adc_c,
                                           target=int(target_adc),
                                           hv_c_old=hv_c, hv_c_new=next_hv_c)
        dec = {"message": msg, "update_state": {"phase": "approving"}}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})
        state["phase"] = "approving"

        # ── 1f: Apply voltage ────────────────────────────────
        cv = {}
        if not done_c:
            cv[f"{tower}C"] = next_hv_c
        if not done_s:
            cv[f"{tower}S"] = next_hv_s
        dec = {"tool": "hv_execute_tool",
               "params": {"command": "voltage", "channel_values": cv},
               "update_state": {"phase": "equalizing"}}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        # voltage 후 상태 업데이트
        state["last_hv_c"] = next_hv_c
        state["last_hv_s"] = next_hv_s
        state["last_suggested_hv_c"] = None
        state["last_suggested_hv_s"] = None
        state["last_adc_c"] = None
        state["last_adc_s"] = None
        state["channel_done_c"] = done_c
        state["channel_done_s"] = done_s
        state["phase"] = "equalizing"

        # ── 1g: Confirmation ─────────────────────────────────
        dec = {"message": MESSAGE_HV_CONFIRM}
        examples.append(make_example(state, history, dec))
        history.append({"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)})
        history.append({"role": "user", "content": "완료"})
        state["phase"] = "idle"

        hv_c = next_hv_c
        hv_s = next_hv_s
        run_number += 1

    return examples


def main():
    output_file = Path(__file__).parent / "data" / "hv_equalization_data.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 타워별 위치 (실제 calculate_position 사용)
    try:
        from tools.position_calculator_tool import calculate_position
        tower_positions = {t: calculate_position(t) or {"x": float(i) * 500, "y": 0.0}
                          for i, t in enumerate(TOWER_ORDER)}
    except Exception:
        tower_positions = {t: {"x": float(i) * 500, "y": 0.0} for i, t in enumerate(TOWER_ORDER)}

    all_ex = []

    # energy, target_adc 모두 패턴마다 랜덤 선택 → 다양성 확보, 데이터 양 절반
    TARGET_ADC_CHOICES = [1000, 1100, 1200, 1300, 1400, 1500]
    ENERGY_CHOICES     = [10, 20, 50, 100, 200]
    events_list        = [500, 1000]

    for tower in TOWER_ORDER:
        pos = tower_positions[tower]
        x, y = pos["x"], pos["y"]
        for events in events_list:
            # 증가 시나리오: 모든 패턴, 각각 랜덤 energy + target
            for pattern in CONVERGENCE_PATTERNS:
                energy     = random.choice(ENERGY_CHOICES)
                target_adc = random.choice(TARGET_ADC_CHOICES)
                all_ex.extend(generate_workflow(
                    tower, x, y, energy, events, target_adc, pattern, going_down=False
                ))
            # 감소 시나리오: 일부 패턴, 각각 랜덤 energy + target
            for pattern in CONVERGENCE_PATTERNS[:3]:
                energy     = random.choice(ENERGY_CHOICES)
                target_adc = random.choice(TARGET_ADC_CHOICES)
                all_ex.extend(generate_workflow(
                    tower, x, y, energy, events, target_adc, pattern, going_down=True
                ))

    # 중간 상태에서 시작하는 샘플 (후반 반복에서도 올바른 결정 학습)
    for tower in TOWER_ORDER:
        pos = tower_positions[tower]
        x, y = pos["x"], pos["y"]
        for events in [1000]:
            for pattern in [CONVERGENCE_PATTERNS[4], CONVERGENCE_PATTERNS[7]]:
                for start_iter in [1, 2]:
                    energy     = random.choice(ENERGY_CHOICES)
                    target_adc = random.choice(TARGET_ADC_CHOICES)
                    all_ex.extend(generate_workflow(
                        tower, x, y, energy, events, target_adc, pattern,
                        going_down=False, start_iter=start_iter
                    ))

    random.shuffle(all_ex)

    with open(output_file, "w", encoding="utf-8") as f:
        for ex in all_ex:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    lengths = [sum(len(m["content"]) for m in ex["messages"]) for ex in all_ex]
    max_chars = max(lengths)
    avg_chars = sum(lengths) / len(lengths)
    print(f"Generated {len(all_ex)} samples -> {output_file}")
    print(f"   char len  max={max_chars:,}  avg={avg_chars:,.0f}  (≈token max={max_chars//2:,}  avg={avg_chars//2:,.0f})")


if __name__ == "__main__":
    main()

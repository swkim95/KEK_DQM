#!/usr/bin/env python3
"""
Energy Scan Agent
다양한 빔 에너지에서 데이터 수집 자동화
"""

import json
import sys
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from datetime import datetime

# 기존 tools import
from tools.daq_tool import DAQRunTool
from tools.position_calculator_tool import calculate_position

from .base_agent import BaseAgent
sys.path.append(str(Path(__file__).parent.parent))
from config import AGENT_MODELS


class EnergyScanAgent(BaseAgent):
    def __init__(
        self,
        energy_config: Dict[float, int],
        tower: str = "T5",
        position: Optional[Dict[str, float]] = None,
        daq_config: str = "setup",
        use_base_model: bool = True,  # Fine-tuning 전에는 base model 사용
        io_handler=None,
    ):
        # Model path 결정
        model_config = AGENT_MODELS["energy_scan"]
        if use_base_model:
            model_path = model_config["base_model"]
            print(f"⚠️  Base model 사용 ({model_path})")
        else:
            fine_tuned_path = Path(model_config["fine_tuned_path"])
            # Fine-tuned 모델 확인
            if fine_tuned_path.exists() and (fine_tuned_path / "config.json").exists():
                model_path = str(fine_tuned_path)
                print(f"✅ Fine-tuned model 사용 ({model_path})")
            else:
                # Fallback to base model
                model_path = model_config["base_model"]
                print(f"⚠️  Fine-tuned 모델 없음. Base model 사용 ({model_path})")
        
        # Base Agent 초기화
        super().__init__(
            model_path=model_path,
            agent_name="EnergyScan",
            io_handler=io_handler,
        )
        
        # Energy 설정 초기화 (빈 dict 허용 - Agent가 자연어로 설정)
        self._init_energy_config = energy_config if energy_config else {}
        
        # Tool 인스턴스 생성
        self.daq_tool = DAQRunTool()
        
        # T5 위치 정보 미리 계산 (Rot 1.5, Tilt 1.0)
        t5_pos = calculate_position("T5") # position_calculator_tool 활용
        if t5_pos:
            self.t5_x = t5_pos['x']
            self.t5_y = t5_pos['y']
        else:
            raise RuntimeError("❌ T5 타워의 위치를 계산할 수 없습니다. Position Scan을 먼저 완료해주세요.")
            
        # State 초기화
        self.state = {
            "phase": "config" if not self._init_energy_config else "idle",  # config → idle → moving_to_center → scanning → completed
            "tower": tower,
            "position": position,
            "daq_config": daq_config,
            
            # Energy 설정
            "energy_config": {
                energy: {
                    "target_events": events,
                    "collected_events": 0,
                    "runs": [],  # [run_number, ...]
                    "completed": False,
                    "completed_at": None
                }
                for energy, events in self._init_energy_config.items()
            },
            
            "scan_order": sorted(list(self._init_energy_config.keys())),
            "current_energy": None,
            "current_energy_idx": 0,
            
            # 세션 정보
            "start_time": datetime.now().isoformat(),
            
            # Plot 설정 (default)
            "plot_method": "PeakADC",  # or "intADC"
            "plot_max_event": None,  # None = all events
        }
        
        self.log(f"Energy Scan Agent 초기화: {list(self._init_energy_config.keys())} GeV")
    
    # ===== System Prompt =====
    
    def _get_system_prompt(self) -> str:
        """System prompt (workflow 정의)"""
        return """You are Energy Scan Agent for test beam experiments.

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
    
    # ===== State Context =====
    
    def _build_state_context(self) -> str:
        """State를 문자열로 변환"""
        lines = []
        
        # 기본 정보
        lines.append(f"Phase: {self.state['phase']}")
        lines.append(f"Tower: {self.state['tower']}")
        lines.append(f"T5 Position: x={self.t5_x:.1f}, y={self.t5_y:.1f}, rot=1.5, tilt=1.0")
        if self.state['position']:
            lines.append(f"Position: {self.state['position']}")
        lines.append("")
        
        # Scan 순서
        lines.append(f"Scan Order: {self.state['scan_order']}")
        lines.append(f"Current Energy: {self.state['current_energy']} GeV (index: {self.state['current_energy_idx']})")
        lines.append("")
        
        # Energy 진행 상황
        lines.append("Energy Progress:")
        for energy in self.state['scan_order']:
            # None 값 건너뛰기
            if energy is None:
                continue
            config = self.state['energy_config'][energy]
            collected = config['collected_events']
            target = config['target_events']
            runs = config['runs']
            completed = config['completed']
            
            if completed:
                status = "✅"
            elif collected > 0:
                status = f"⏳ {collected}/{target}"
            else:
                status = "⏸️  Not started"
            
            run_info = f"Runs: {runs}" if runs else ""
            
            # 현재 에너지 표시
            if energy == self.state['current_energy']:
                lines.append(f"  {energy} GeV: {status} {run_info} ← CURRENT (target: {target} events)")
            else:
                lines.append(f"  {energy} GeV: {status} {run_info}")
        
        return "\n".join(lines)
    
    def _get_step_hint(self) -> str:
        """현재 상태 요약 - AI가 학습을 통해 다음 단계를 스스로 결정"""
        phase = self.state.get("phase", "config")
        current_energy = self.state.get("current_energy")
        scan_order = self.state.get("scan_order", [])
        idx = scan_order.index(current_energy) + 1 if current_energy in scan_order else 0
        total = len(scan_order)
        return f"Phase: {phase} | Energy: {current_energy} GeV ({idx}/{total})"

    def build_full_context(self, current_input: Optional[str] = None) -> str:
        """전체 context 생성 (단계별 힌트 포함)"""
        
        # 마지막 사용자 입력이 history에 있는데 아직 current_input으로 처리되지 않은 경우 처리
        # (EM_data_gen.py의 첫 번째 턴 학습 데이터와 형식을 맞추기 위함)
        if current_input is None and self.conversation_history:
            # 마지막 메시지가 user인 경우 current_input으로 사용
            if self.conversation_history[-1]["role"] == "user":
                current_input = self.conversation_history[-1]["content"]
                # history에서는 제외하고 보여주기 위해 임시 history 생성
                temp_history = self.conversation_history[:-1]
            else:
                temp_history = self.conversation_history
        else:
            temp_history = self.conversation_history

        parts = []
        parts.append("=== Current State ===")
        parts.append(self._build_state_context())
        parts.append("")
        parts.append("=== Recent Conversation ===")
        # 임시 history로 context 생성
        history_lines = []
        if not temp_history:
            history_lines.append("(No conversation yet)")
        else:
            recent_history = temp_history[-10:]
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Agent"
                content = msg["content"]
                if content == "완료" or content.strip() == "완료":
                    history_lines.append(f"{role}: 완료 [IMPORTANT: User confirmed completion]")
                else:
                    history_lines.append(f"{role}: {content}")
        parts.append("\n".join(history_lines))
        parts.append("")
        
        if current_input:
            parts.append("=== Current User Input ===")
            parts.append(current_input)
            parts.append("")
        
        parts.append("=== Your Task ===")
        parts.append(self._get_step_hint())
        parts.append("")
        parts.append("Output JSON with tool name and parameters.")
        return "\n".join(parts)
    
    # ===== Main Loop =====
    
    def _print_summary(self):
        """현재 진행 상황 요약 출력 (Dash보드 스타일)"""
        print(f"\n📊 Energy Scan Progress Summary:")
        print("-" * 70)
        tower = self.state.get('tower', 'T5')
        print(f"Tower: {tower} | Position: x={self.t5_x:.1f}, y={self.t5_y:.1f}")
        print("-" * 70)
        
        for energy in self.state['scan_order']:
            if energy is None: continue
            config = self.state['energy_config'][energy]
            completed = config.get('completed', False)
            collected = config.get('collected_events', 0)
            target = config.get('target_events', 0)
            mark = "✅" if completed else ("➡️ " if energy == self.state['current_energy'] else "  ")
            status_text = "Completed" if completed else f"{collected}/{target} events"
            run_info = f" (Runs: {config['runs']})" if config['runs'] else ""
            print(f"  {mark} {energy} GeV: {status_text}{run_info}")
        print("-" * 70)

    def run(self):
        """
        Energy Scan 실행 (대화형)
        """
        print(f"\n{'='*70}")
        print(f"⚡ Energy Scan Agent Started")
        print(f"{'='*70}")
        
        self.log("Energy Scan 시작")
        
        # 모델이 로드되어 있는지 확인
        if self.model is None:
            raise RuntimeError("Model not loaded. Use 'with agent:' statement.")
        
        while True:
            try:
                # 진행 상황 요약 출력
                self._print_summary()

                # scanning 중 current_energy가 None이면 첫 번째 미완료 에너지로 파생
                # (data gen은 루프 시작 시 current_energy를 미리 세팅하므로 동작 방식 일치 필요)
                if self.state.get('phase') == 'scanning' and self.state.get('current_energy') is None:
                    for _e in self.state['scan_order']:
                        if not self.state['energy_config'][_e].get('completed', False):
                            self.state['current_energy'] = _e
                            self.state['current_energy_idx'] = self.state['scan_order'].index(_e)
                            break

                # 1. 현재 context 생성
                context = self.build_full_context()
                
                # 2. Agent 의사결정
                decision = self.decide(context)
                
                # DEBUG: Agent의 의사결정 시각화
                print(f"\n🔍 Agent Decision:")
                print(json.dumps(decision, indent=2, ensure_ascii=False))
                
                # 3. 에러 체크
                if "error" in decision:
                    print(f"\n❌ Agent Error: {decision['error']}")
                    self.log(f"Error: {decision}")
                    break
                
                # 4. State 업데이트
                if "update_state" in decision:
                    before = sum(1 for c in self.state["energy_config"].values() if c.get("completed"))
                    self._update_state(decision["update_state"])
                    after = sum(1 for c in self.state["energy_config"].values() if c.get("completed"))
                    if after > before:
                        self.io.send_ai_message(self._format_progress())

                # 5. Message 또는 Tool 실행
                message = decision.get("message")
                tool_name = decision.get("tool")

                if message:
                    self.io.send_ai_message(message)
                    self.add_to_history("assistant", json.dumps(decision, ensure_ascii=False))
                    if "모든 에너지 스캔이 완료되었습니다" in message:
                        self.log("모든 에너지 스캔 완료 - 자동 종료")
                        break
                    user_input = self.io.get_input()
                    if user_input in ["종료", "exit"]: break
                    self.add_to_history("user", user_input)
                    continue

                if tool_name and tool_name != "none":
                    params = decision.get("params", {})
                    print(f"\n🤖 Executing Tool: {tool_name}")
                    result = self._execute_tool(tool_name, params)
                    print(f"📝 Tool Result: {result[:200]}...")
                    self.add_to_history("assistant", json.dumps(decision, ensure_ascii=False))
                    continue

                if "update_state" in decision:
                    self.add_to_history("assistant", json.dumps(decision, ensure_ascii=False))
                    continue

                if all(c.get("completed", False) for c in self.state["energy_config"].values()) and self.state["energy_config"]:
                    print("\n✅ 모든 에너지 스캔이 완료되었습니다.")
                    break

                break
            except KeyboardInterrupt: break
            except Exception as e:
                print(f"\n❌ 오류 발생: {str(e)}")
                break

    def _reset_state_for_new_scan(self):
        """새로운 스캔을 위해 상태 초기화"""
        self.state.update({
            "phase": "config",
            "energy_config": {},
            "scan_order": [],
            "current_energy": None,
            "current_energy_idx": 0,
            "start_time": datetime.now().isoformat()
        })
        self.conversation_history = [] # 대화 내역도 초기화하여 꼬임 방지
        self.log("상태 초기화 완료 (새로운 스캔)")
    
    # ===== Tool 실행 =====
    
    def _execute_tool(self, tool_name: str, params: Dict) -> str:
        """Tool 실행"""
        print(f"\n🤖 Agent Decision:")
        print(f"   Tool: {tool_name}")
        print(f"   Params: {json.dumps(params, ensure_ascii=False)}")
        print()
        
        if tool_name == "none":
            # Tool 없음 - state만 업데이트
            return "no_tool_executed"
        
        elif tool_name == "daq_run_tool":
            # Determine the correct energy (don't blindly trust LLM's beam_energy)
            energy_key = params.get('beam_energy')
            if energy_key is not None:
                try:
                    energy_key = int(float(energy_key))
                except Exception:
                    energy_key = None
            # Reject if LLM gave an energy not in our config
            if energy_key is not None and energy_key not in self.state['energy_config']:
                self.log(f"WARNING: LLM beam_energy={energy_key} not in config — falling back")
                energy_key = None
            if energy_key is None:
                energy_key = self.state.get('current_energy')
            if energy_key is None:
                for e in self.state['scan_order']:
                    if not self.state['energy_config'][e].get('completed', False):
                        energy_key = e
                        break

            # Override both events AND beam_energy with authoritative Python values
            if energy_key is not None and energy_key in self.state['energy_config']:
                params['events'] = self.state['energy_config'][energy_key]['target_events']
                params['beam_energy'] = energy_key

            params['program'] = "EM Scan"

            # DAQ 실행. daq_tool 내부의 dqm_session.start()이 monit --LIVE를 띄워서
            # DAQ 동안 우측 하단 DQM 패널이 실시간 갱신된다 — 여기가 유일한 플롯 경로.
            result = self.daq_tool.execute(params, line_callback=self.io.send_tool_output)

            # Run number 추출 및 state 업데이트
            run_number = self._extract_run_number(result)
            if run_number:
                self.state['last_run_number'] = run_number
                # energy_key is already resolved above
                if energy_key is not None and energy_key in self.state['energy_config']:
                    self.state['current_energy'] = energy_key
                    self.state['energy_config'][energy_key]['runs'].append(run_number)
                    self.state['energy_config'][energy_key]['collected_events'] = params.get('events', 0)
                    self.log(f"DAQ Run {run_number} 완료: {energy_key} GeV, {params.get('events', 0)} events")

            return result

        else:
            print(f"⚠️  Unknown tool: {tool_name}")
            self.log(f"Unknown tool: {tool_name}")
            return f"Error: Unknown tool {tool_name}"
    
    # ===== Helper 함수 =====
    
    # Fields the LLM must not overwrite (set once by Python at init)
    _PROTECTED_FIELDS = frozenset({
        "tower", "daq_config", "start_time", "plot_method", "plot_max_event",
    })

    def _update_state(self, updates: Dict[str, Any]):
        """State 업데이트 (energy_config는 deep merge로 기존 필드 보존)"""
        for key, value in updates.items():
            if key == "energy_config" and isinstance(value, dict):
                for energy_key, config_value in value.items():
                    try:
                        int_key = int(energy_key)
                    except:
                        int_key = energy_key
                    if int_key in self.state['energy_config'] and isinstance(config_value, dict):
                        # 부분 업데이트: 기존 필드를 보존하고 제공된 필드만 덮어씀
                        # target_events는 Python 파서가 설정한 값 — LLM이 변경 불가
                        safe_update = {k: v for k, v in config_value.items() if k != 'target_events'}
                        # completed=True는 실제 런이 기록된 경우에만 허용
                        current_runs = self.state['energy_config'][int_key].get('runs', [])
                        if safe_update.get('completed') and not current_runs:
                            safe_update.pop('completed', None)
                            self.log(f"WARNING: LLM tried to mark {int_key} GeV complete without runs — rejected")
                        self.state['energy_config'][int_key].update(safe_update)
                        # completed=True 시 완료 시각 자동 기록
                        if self.state['energy_config'][int_key].get('completed') and not self.state['energy_config'][int_key].get('completed_at'):
                            self.state['energy_config'][int_key]['completed_at'] = datetime.now().strftime("%H:%M:%S")
                    else:
                        # 새 에너지 항목 전체 생성
                        self.state['energy_config'][int_key] = config_value
                self.state['scan_order'] = sorted(
                    [e for e in self.state['energy_config'].keys() if e is not None]
                )
                self.log(f"State updated: energy_config (deep merge), scan_order={self.state['scan_order']}")

            elif key == "current_energy" and value is not None:
                try:
                    self.state[key] = int(value)
                except:
                    self.state[key] = value
                self.log(f"State updated: {key} = {self.state[key]}")

            elif key in self._PROTECTED_FIELDS:
                self.log(f"WARNING: LLM tried to update protected field '{key}' = {value} — rejected")
            else:
                self.state[key] = value
                self.log(f"State updated: {key} = {value}")
    
    def _extract_run_number(self, daq_output: str = None) -> Optional[int]:
        """
        Run number 추출
        runnum.txt 파일에서 읽기
        """
        try:
            from tools.config_loader import get_path_config
            runnum_file = Path(get_path_config("RunNumberFile"))
            if runnum_file.exists():
                with open(runnum_file, 'r') as f:
                    # Run이 종료된 후 runnum.txt가 다음 번호로 업데이트되므로, 
                    # 방금 종료된 Run 정보를 위해 -1을 수행함
                    val = f.read().strip()
                    run_number = int(val) - 1
                    return run_number
        except Exception as e:
            self.log(f"Run number 읽기 실패: {e}")
        
        # Fallback: DAQ output에서 추출
        if daq_output:
            import re
            match = re.search(r'Run:?\s*(\d+)', daq_output)
            if match:
                return int(match.group(1))
        
        return None
    
    def _check_completion(self) -> bool:
        """전체 scan 완료 확인 (개별 에너지 완료는 모델이 판단)"""
        if not self.state['energy_config']:
            return False
        
        # 모든 에너지가 completed 상태인지 확인
        return all(config.get('completed', False) for config in self.state['energy_config'].values())
    
    def _format_progress(self) -> str:
        """현재 진행 상황을 문자열로 반환 (AI 메시지용)"""
        total = len(self.state['scan_order'])
        done = sum(1 for c in self.state['energy_config'].values() if c.get('completed'))
        lines = [
            f"📊 Energy Scan  —  {done} / {total} 완료",
            f"T5 위치:  x = {self.t5_x:.1f},  y = {self.t5_y:.1f}",
            "─" * 36,
        ]
        for energy in self.state['scan_order']:
            if energy is None:
                continue
            config = self.state['energy_config'].get(energy, {})
            if config.get('completed'):
                runs_str = ', '.join(str(r) for r in config['runs']) if config['runs'] else '-'
                lines.append(f"  ✅  {energy} GeV   {config['target_events']} events   Run {runs_str}")
            else:
                lines.append(f"       {energy} GeV   {config.get('target_events',0)} events")
        lines.append("─" * 36)
        return "\n".join(lines)

    def _print_summary(self):
        """완료 요약 출력 (CLI용)"""
        print("\n📊 Energy Scan Summary:")
        print("-" * 70)
        for energy in sorted(self.state['scan_order']):
            config = self.state['energy_config'][energy]
            print(f"  {energy} GeV: {config['collected_events']}/{config['target_events']} events")
            print(f"    Runs: {config['runs']}")
        print("-" * 70)

        elapsed = (datetime.now() - datetime.fromisoformat(self.state['start_time'])).total_seconds() / 60
        print(f"Total time: {elapsed:.1f} minutes")
    
    # ===== State 접근 =====
    
    def get_state_summary(self) -> Dict[str, Any]:
        """현재 상태 요약 반환 (API용)"""
        completed_count = sum(
            1 for config in self.state['energy_config'].values()
            if config['completed']
        )
        
        return {
            "phase": self.state['phase'],
            "current_energy": self.state['current_energy'],
            "completed_count": completed_count,
            "total_energies": len(self.state['scan_order']),
            "progress_percent": (completed_count / len(self.state['scan_order'])) * 100,
            "energy_config": self.state['energy_config']
        }

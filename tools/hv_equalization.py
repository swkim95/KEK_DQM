#!/usr/bin/env python3
"""
HV Equalization Advisor System - Exponential Fitting
물리학 기반 HV-ADC exponential 관계 모델링 및 제안 시스템
"""

import subprocess
import re
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple
from langchain_core.tools import tool

try:
    from scipy.optimize import curve_fit
    FITTING_AVAILABLE = True
except ImportError:
    FITTING_AVAILABLE = False

# peakADC_avg.py 경로
PEAK_ADC_AVG_SCRIPT = "/Users/yhep/DRC/KEK/peak/peakADC_avg.py"
DEFAULT_TARGET_ADC = 1800.0

class ExponentialHVPredictor:
    """물리학 기반 Exponential HV-ADC 관계 예측기"""
    
    def __init__(self):
        self.data_points = {"C": [], "S": []}
        self.coefficients = {"C": None, "S": None}
        self.is_fitted = {"C": False, "S": False}
        self.channel_status = {"C": "active", "S": "active"}
        self.fit_method = {"C": None, "S": None}
        
    def reset_data(self, message="데이터 초기화됨"):
        """모든 데이터 초기화"""
        self.data_points = {"C": [], "S": []}
        self.coefficients = {"C": None, "S": None}
        self.is_fitted = {"C": False, "S": False}
        self.channel_status = {"C": "active", "S": "active"}
        self.fit_method = {"C": None, "S": None}
    
    def mark_channel_done(self, channel):
        """채널 완료 표시 - 해당 채널은 더 이상 suggestion 안함"""
        self.channel_status[channel] = "done"
    
    def is_channel_active(self, channel):
        """채널이 아직 active 상태인지 확인"""
        return self.channel_status[channel] == "active"
            
    def exponential_func(self, hv, A, B):
        """Exponential function: ADC = A * exp(B * HV)"""
        return A * np.exp(B * hv)
    
    def _fit_exponential(self, channel):
        """채널별 exponential fitting"""
        points = self.data_points[channel]
        if len(points) < 2:
            return False, None
        
        hvs = [p[0] for p in points]
        adcs = [p[1] for p in points]
        
        try:
            fitting_message = None
            
            if FITTING_AVAILABLE and len(points) >= 3:
                # scipy를 이용한 curve fitting
                popt, _ = curve_fit(self.exponential_func, hvs, adcs, 
                                  p0=[100, 0.001],  # 초기 추정값
                                  maxfev=1000)
                self.coefficients[channel] = popt
                self.is_fitted[channel] = True
                self.fit_method[channel] = "scipy_curve_fit"
                fitting_message = f"{channel} 채널 fitting: ADC = {popt[0]:.2f} * exp({popt[1]:.5f} * HV)"
            elif len(points) >= 2:
                # 간단한 2점 기반 fitting
                hv1, adc1 = points[0]
                hv2, adc2 = points[-1]
                
                if hv1 != hv2 and adc1 > 0 and adc2 > 0:
                    # ADC = A * exp(B * HV)에서 B = ln(adc2/adc1) / (hv2-hv1)
                    B = math.log(adc2 / adc1) / (hv2 - hv1)
                    A = adc1 / math.exp(B * hv1)
                    
                    self.coefficients[channel] = (A, B)
                    self.is_fitted[channel] = True
                    self.fit_method[channel] = "two_point"
                    fitting_message = f"{channel} 채널 fitting: ADC = {A:.2f} * exp({B:.5f} * HV)"
            
            return self.is_fitted[channel], fitting_message
            
        except Exception as e:
            error_msg = f"❌ {channel} 채널 fitting 실패: {e}"
            return False, error_msg
    
    def predict_hv_for_target(self, channel, target_adc):
        """Target ADC에 필요한 HV 예측"""
        if not self.is_fitted[channel] or self.coefficients[channel] is None:
            return None
        
        A, B = self.coefficients[channel]
        try:
            # ADC = A * exp(B * HV) => HV = ln(ADC/A) / B
            if A > 0 and B != 0:
                required_hv = math.log(target_adc / A) / B
                return max(0, required_hv)  # HV는 양수
        except (ValueError, ZeroDivisionError):
            pass
        
        return None
    
    def initial_exploration(self, channel, current_hv, current_adc, target_adc):
        """초기 ±10V 탐색으로 방향성 파악"""
        
        hv_step = 10
        # 방향성 판단: target보다 크면 아래로, 작으면 위로
        if current_adc > target_adc:
            # ADC가 너무 높음 → HV 낮춰야 함
            explore_hv = current_hv - hv_step
            direction = "down"
        else:
            # ADC가 너무 낮음 → HV 높여야 함  
            explore_hv = current_hv + hv_step
            direction = "up"
        
        # 현재점을 데이터에 추가
        self.data_points[channel].append((current_hv, current_adc))
        
        return {
            "suggested_hv": explore_hv,
            "direction": direction,
            "reason": f"ADC {current_adc:.1f} {'>' if current_adc > target_adc else '<'} target {target_adc}, {direction} 방향으로 {hv_step}V 이동"
        }
    
    def add_data_point(self, channel, hv, adc):
        """새로운 (HV, ADC) 점 추가 및 exponential re-fitting"""
        self.data_points[channel].append((hv, adc))
        
        # 점이 2개 이상이면 fitting 시도
        fitting_message = None
        if len(self.data_points[channel]) >= 2:
            fitted, fitting_message = self._fit_exponential(channel)
            
        return fitting_message  # fitting 결과 메시지 반환
    
    def predict_parallel_hv_adjustment(self, current_hv_c, current_adc_c, target_adc_c,
                                     current_hv_s, current_adc_s, target_adc_s):
        """Exponential fitting 기반 C, S 채널 병렬 HV 조정 예측"""
        
        result = {"C": None, "S": None}
        
        # C 채널 예측 (done 상태면 skip)
        if not self.is_channel_active("C"):
            result["C"] = {
                "status": "done",
                "message": "C 채널 equalization 완료됨 - suggestion 없음",
                "current_adc": current_adc_c,
                "target_adc": target_adc_c,
                "adc_error": abs(target_adc_c - current_adc_c)
            }
        elif self.is_fitted["C"]:
            # Exponential fitting으로 정확한 HV 예측
            required_hv = self.predict_hv_for_target("C", target_adc_c)
            if required_hv is not None:
                hv_change = int(round(required_hv - current_hv_c))
                hv_change = max(-200, min(200, hv_change))  # 안전 제한
                
                result["C"] = {
                    "hv_change": hv_change,
                    "current_hv": current_hv_c,
                    "target_hv": current_hv_c + hv_change,
                    "current_adc": current_adc_c,
                    "target_adc": target_adc_c,
                    "adc_error": abs(target_adc_c - current_adc_c),
                    "method": "exponential_fit"
                }
            else:
                result["C"] = {"error": "Exponential fitting 계산 실패"}
        else:
            # 초기 탐색 제안
            exploration = self.initial_exploration("C", current_hv_c, current_adc_c, target_adc_c)
            hv_change = exploration["suggested_hv"] - current_hv_c
            
            result["C"] = {
                "hv_change": hv_change,
                "current_hv": current_hv_c,
                "target_hv": exploration["suggested_hv"],
                "current_adc": current_adc_c,
                "target_adc": target_adc_c,
                "adc_error": abs(target_adc_c - current_adc_c),
                "method": "initial_exploration",
                "reason": exploration["reason"]
            }
        
        # S 채널 예측 (done 상태면 skip)
        if not self.is_channel_active("S"):
            result["S"] = {
                "status": "done",
                "message": "S 채널 equalization 완료됨 - suggestion 없음",
                "current_adc": current_adc_s,
                "target_adc": target_adc_s,
                "adc_error": abs(target_adc_s - current_adc_s)
            }
        elif self.is_fitted["S"]:
            required_hv = self.predict_hv_for_target("S", target_adc_s)
            if required_hv is not None:
                hv_change = int(round(required_hv - current_hv_s))
                hv_change = max(-200, min(200, hv_change))
                
                result["S"] = {
                    "hv_change": hv_change,
                    "current_hv": current_hv_s,
                    "target_hv": current_hv_s + hv_change,
                    "current_adc": current_adc_s,
                    "target_adc": target_adc_s,
                    "adc_error": abs(target_adc_s - current_adc_s),
                    "method": "exponential_fit"
                }
            else:
                result["S"] = {"error": "Exponential fitting 계산 실패"}
        else:
            exploration = self.initial_exploration("S", current_hv_s, current_adc_s, target_adc_s)
            hv_change = exploration["suggested_hv"] - current_hv_s
            
            result["S"] = {
                "hv_change": hv_change,
                "current_hv": current_hv_s,
                "target_hv": exploration["suggested_hv"],
                "current_adc": current_adc_s,
                "target_adc": target_adc_s,
                "adc_error": abs(target_adc_s - current_adc_s),
                "method": "initial_exploration",
                "reason": exploration["reason"]
            }
        
        return result
    
    def add_measurement_for_learning(self, channel, current_hv, current_adc, target_adc, 
                                   applied_change, final_adc):
        """새 측정 결과로 exponential fitting 업데이트"""
        
        final_hv = current_hv + applied_change
        
        # 새로운 점들 추가
        self.add_data_point(channel, current_hv, current_adc)
        fitting_message = self.add_data_point(channel, final_hv, final_adc)
        
        return fitting_message, ""

class HVEqualizationSession:
    """HV Equalization 세션 관리 클래스 (Exponential Fitting 지원)"""
    
    def __init__(self):
        self.sessions = {}  # session_id -> session_data
        self.exp_predictor = ExponentialHVPredictor()
        self.exp_initialized = False
    
    def reset_for_new_pmt(self, session_id: str):
        """새로운 PMT를 위한 데이터 초기화 (target 유지)"""
        if session_id in self.sessions:
            self.exp_predictor.reset_data()
            
            session = self.sessions[session_id]
            target_c = session.get("target_c")
            target_s = session.get("target_s")
            
            self.sessions[session_id] = {
                "target_c": target_c,
                "target_s": target_s,
                "history": [],
                "active": True,
                "current_hv_c": 800,
                "current_hv_s": 800,
                "iteration": 0,
                "center": None
            }
            
            return f"""새로운 PMT equalization 준비 완료!

Target 유지: C={target_c} ADC, S={target_s} ADC
데이터 초기화: 이전 exponential fitting 데이터 모두 지워짐

사용법: "run 12345 hv_c=800 hv_s=800"부터 새로 시작하세요."""
    
    def start_session(self, session_id: str, target_c: float, target_s: float) -> str:
        """새로운 HV equalization 세션 시작 (Exponential Fitting)"""
        
        if not self.exp_initialized:
            self.exp_predictor.reset_data()
            self.exp_initialized = True
            exp_status = "새로운 세션 시작 - 실시간 exponential fitting (±10V 탐색부터)"
        else:
            exp_status = "세션 진행 중 - 실시간 exponential fitting"
        
        self.sessions[session_id] = {
            "target_c": target_c,
            "target_s": target_s,
            "history": [],
            "active": True,
            "current_hv_c": 800,
            "current_hv_s": 800,
            "iteration": 0,
            "center": None
        }
        
        return f"""Exponential Fitting HV Equalization 세션 시작!

Target: C={target_c} ADC, S={target_s} ADC
{exp_status}

사용법:
  1) "center T1" (센터 필수 입력, 이미 완료했다면 생략 가능)
  2) 첫 run: "run 12345 hv_c=800 hv_s=800" (HV 명시 필요, 방향성 ±10V 탐색 시작)
  3) 이후: "run 12346" (AI 제안대로 설정했다고 가정) 
  4) 수동 조정: "run 12347 hv_c=805 hv_s=815" (직접 조정한 경우만)"""
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 정보 조회"""
        return self.sessions.get(session_id)
    
    def process_parallel_run(self, session_id: str, run_num: int, adc_c: float, adc_s: float,
                           hv_c: int, hv_s: int, hv_source: str = "user_explicit") -> str:
        """Exponential fitting을 사용한 병렬 run 처리 (자동 학습 포함)"""
        
        if session_id not in self.sessions:
            return "ERROR: 세션이 시작되지 않았습니다. 먼저 target을 설정하세요."
        
        session = self.sessions[session_id]
        session["iteration"] += 1
        
        fitting_updates = []
        if "last_suggestion" in session and hv_source == "ai_suggestion":
            auto_learn_messages = self._auto_learn_from_previous_suggestion(session, hv_c, hv_s, adc_c, adc_s)
            fitting_updates.extend(auto_learn_messages)
        
        session["current_hv_c"] = hv_c
        session["current_hv_s"] = hv_s
        
        target_c = session["target_c"]
        target_s = session["target_s"]
        
        predictions = self.exp_predictor.predict_parallel_hv_adjustment(
            hv_c, adc_c, target_c,
            hv_s, adc_s, target_s
        )
        
        result_c = predictions["C"]
        result_s = predictions["S"]
        
        if "error" in result_c:
            return f"ERROR: {result_c['error']}"
        if "error" in result_s:
            return f"ERROR: {result_s['error']}"
        if result_c.get("status") == "done":
            change_c = 0
            next_hv_c = hv_c
        else:
            change_c = result_c["hv_change"]
            next_hv_c = hv_c + change_c
            
        if result_s.get("status") == "done":
            change_s = 0
            next_hv_s = hv_s
        else:
            change_s = result_s["hv_change"]
            next_hv_s = hv_s + change_s
        
        session["history"].append({
            "run": run_num,
            "hv_c": hv_c, "hv_s": hv_s,
            "adc_c": adc_c, "adc_s": adc_s,
            "change_c": change_c, "change_s": change_s
        })
        
        error_c = abs(adc_c - target_c)
        error_s = abs(adc_s - target_s)
        session["last_suggestion"] = {
            "run_num": run_num,
            "suggested_hv_c": next_hv_c,
            "suggested_hv_s": next_hv_s,
            "suggested_change_c": change_c, 
            "suggested_change_s": change_s,
            "from_hv_c": hv_c,
            "from_hv_s": hv_s,
            "from_adc_c": adc_c,
            "from_adc_s": adc_s,
            "target_c": target_c,
            "target_s": target_s
        }
        
        if hv_source == "ai_suggestion":
            hv_info = "AI 제안대로 HV 설정됨"
        elif hv_source == "user_explicit":
            hv_info = "사용자가 직접 HV 설정함"
        else:
            hv_info = ""
        result = f"""Run {run_num} Analysis: {hv_info}

Current ADC:
  C: {adc_c:.1f} (target: {target_c}, error: ±{error_c:.1f})
  S: {adc_s:.1f} (target: {target_s}, error: ±{error_s:.1f})

HV Adjustment:"""
        
        if result_c.get("status") == "done":
            result += f"\n  C: {hv_c}V → DONE (equalization 완료)"
        else:
            result += f"\n  C: {hv_c}V → {change_c:+d}V → {next_hv_c}V"
            
        if result_s.get("status") == "done":
            result += f"\n  S: {hv_s}V → DONE (equalization 완료)"
        else:
            result += f"\n  S: {hv_s}V → {change_s:+d}V → {next_hv_s}V"
        
        exploration_notes = []
        if result_c.get("method") == "initial_exploration":
            exploration_notes.append(f"C 채널: {result_c.get('reason')}")
        if result_s.get("method") == "initial_exploration":
            exploration_notes.append(f"S 채널: {result_s.get('reason')}")
        if exploration_notes:
            result += "\n\n초기 탐색 가이드:"
            for note in exploration_notes:
                result += f"\n  - {note}"
        if fitting_updates:
            result += f"\n\nExponential Fitting Updates:"
            for update in fitting_updates:
                result += f"\n  {update}"
        
        fitting_summary = self._format_fitting_summary()
        if fitting_summary:
            result += "\n\nFitting Model:"
            result += f"\n{fitting_summary}"
        
        return result
    
    def _auto_learn_from_previous_suggestion(self, session: dict, current_hv_c: int, current_hv_s: int, 
                                           current_adc_c: float, current_adc_s: float):
        """이전 AI 제안이 적용되었는지 확인하고 자동 학습"""
        
        messages = []
        
        last_suggestion = session["last_suggestion"]
        suggested_hv_c = last_suggestion["suggested_hv_c"] 
        suggested_hv_s = last_suggestion["suggested_hv_s"]
        
        hv_tolerance = 2
        if abs(current_hv_c - suggested_hv_c) <= hv_tolerance:
            try:
                fitting_msg, _ = self.exp_predictor.add_measurement_for_learning(
                    "C", 
                    last_suggestion["from_hv_c"], 
                    last_suggestion["from_adc_c"],
                    last_suggestion["target_c"],
                    last_suggestion["suggested_change_c"],
                    current_adc_c
                )
                if fitting_msg:
                    messages.append(fitting_msg)
                    
            except Exception as e:
                pass
        
        if abs(current_hv_s - suggested_hv_s) <= hv_tolerance:
            try:
                fitting_msg, _ = self.exp_predictor.add_measurement_for_learning(
                    "S",
                    last_suggestion["from_hv_s"],
                    last_suggestion["from_adc_s"], 
                    last_suggestion["target_s"],
                    last_suggestion["suggested_change_s"],
                    current_adc_s
                )
                if fitting_msg:
                    messages.append(fitting_msg)
                    
            except Exception as e:
                pass
        
        session.pop("last_suggestion", None)
        
        return messages
    
    def _format_fitting_summary(self) -> str:
        """채널별 fitting 식과 알고리즘 정보를 문자열로 생성"""
        predictor = self.exp_predictor
        method_labels = {
            "scipy_curve_fit": "scipy curve_fit",
            "two_point": "2점 추정"
        }
        lines = []
        for channel in ["C", "S"]:
            if predictor.is_fitted[channel] and predictor.coefficients[channel] is not None:
                A, B = predictor.coefficients[channel]
                method_label = method_labels.get(predictor.fit_method.get(channel), "미지정 알고리즘")
                point_count = len(predictor.data_points[channel])
                lines.append(
                    f"  {channel}: ADC = {A:.2f} * exp({B:.5f} * HV) "
                    f"[{method_label}, 데이터 {point_count}점]"
                )
        return "\n".join(lines)
    
    def add_measurement_for_learning(self, session_id: str, channel: str, hv_before: int, 
                                   adc_before: float, hv_after: int, adc_after: float, target_adc: float):
        """실제 측정 결과로 exponential fitting 업데이트"""
        try:
            applied_change = hv_after - hv_before
            fitting_msg, _ = self.exp_predictor.add_measurement_for_learning(
                channel, hv_before, adc_before, target_adc, applied_change, adc_after
            )
            
            result = f"✅ {channel} 채널 데이터 추가 완료!"
            if fitting_msg:
                result += f"\n{fitting_msg}"
            
            return result
            
        except Exception as e:
            return f"exponential fitting 업데이트 실패: {e}"
            

# 전역 세션 매니저
_session_manager = HVEqualizationSession()

def run_peak_adc_analysis(run_num: int, center: str) -> Tuple[Optional[float], Optional[float]]:
    """peakADC_avg.py를 실행하여 C, S 평균값을 가져옴"""
    try:
        if not center or not str(center).strip():
            return None, None
        
        center_arg = str(center).strip()
        # peakADC_avg.py 실행
        result = subprocess.run(
            ["python3", PEAK_ADC_AVG_SCRIPT, "--run", str(run_num), "--center", center_arg],
            capture_output=True,
            text=True,
            cwd="/Users/yhep/DRC/KEK"
        )
        
        if result.returncode != 0:
            return None, None
        
        # 출력 파싱 (C: 85.23\nS: 140.56 형식)
        output_lines = result.stdout.strip().split('\n')
        c_value = None
        s_value = None
        
        for line in output_lines:
            line = line.strip()
            if line.startswith('C:'):
                try:
                    c_value = float(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('S:'):
                try:
                    s_value = float(line.split(':')[1].strip())
                except:
                    pass
        
        return c_value, s_value
        
    except Exception as e:
        return None, None


@tool
def hveq_center_init_tool(center_input: str) -> str:
    """
    Center 정보를 설정하고 기본 target(1800)으로 세션을 초기화합니다.
    
    Args:
        center_input: "center T1" 또는 "center=T1" 형식의 문자열
    """
    try:
        center_match = re.search(r'center\s*(?:=|:)?\s*([A-Za-z0-9_-]+)', center_input, re.IGNORECASE)
        if not center_match:
            return "ERROR: Center 정보를 찾을 수 없습니다.\n예: 'center T1'"
        
        center_value = center_match.group(1).strip().upper()
        session_id = "default"
        start_message = _session_manager.start_session(session_id, DEFAULT_TARGET_ADC, DEFAULT_TARGET_ADC)
        session = _session_manager.get_session(session_id)
        if session:
            session["center"] = center_value
        
        guide = (
            f"Center {center_value} 세션이 준비되었습니다.\n"
            f"- Target C/S = {DEFAULT_TARGET_ADC} ADC (기본값)\n"
            "- 다음 단계: 'run 12345 hv_c=300 hv_s=400'처럼 첫 run을 입력하면 방향성(±10V) 탐색을 시작합니다."
        )
        return f"{guide}\n\n{start_message}"
    except Exception as e:
        return f"ERROR: Center 초기화 실패: {str(e)}"


@tool
def hveq_session_start_tool(target_input: str) -> str:
    """
    HV Equalization 세션을 시작합니다.
    
    Args:
        target_input: Target 값 (예: "target C=1500 S=1500")
    
    Returns:
        세션 시작 확인 메시지
    """
    try:
        # Target 값 파싱
        target_c = None
        target_s = None
        
        # "C=100" 패턴 찾기
        c_match = re.search(r'C\s*=\s*(\d+(?:\.\d+)?)', target_input, re.IGNORECASE)
        if c_match:
            target_c = float(c_match.group(1))
        
        # "S=150" 패턴 찾기
        s_match = re.search(r'S\s*=\s*(\d+(?:\.\d+)?)', target_input, re.IGNORECASE)
        if s_match:
            target_s = float(s_match.group(1))
        
        default_notes = []
        if target_c is None:
            target_c = DEFAULT_TARGET_ADC
            default_notes.append("C")
        if target_s is None:
            target_s = DEFAULT_TARGET_ADC
            default_notes.append("S")
        
        # 세션 시작 (기본 세션 ID 사용)
        session_id = "default"
        result = _session_manager.start_session(session_id, target_c, target_s)
        if default_notes:
            channels = ", ".join(default_notes)
            result += f"\n\n⚠️ Target {channels} 값이 입력되지 않아 기본값 1800을 사용했습니다."
        
        return result
        
    except Exception as e:
        return f"ERROR: HV Equalization 세션 시작 실패: {str(e)}"

@tool
def hveq_analyze_suggest_tool(run_input: str) -> str:
    """
    Exponential Fitting을 사용한 병렬 Run 분석 및 HV 조정 제안
    
    Args:
        run_input: "run 12345" (AI 제안대로) 또는 "run 12345 hv_c=800 hv_s=800" (직접 설정)
                   Center 정보는 사전에 "center T1" 명령으로 등록하거나 run 입력에 "center T1"을 포함할 수 있습니다.
    
    Returns:
        물리학 기반 exponential 모델로 HV 조정 제안
        - HV 값 생략 시: 이전 AI 제안대로 설정했다고 가정
        - HV 값 명시 시: 사용자가 직접 조정한 값 사용
    """
    try:
        # Run 번호 파싱
        run_match = re.search(r'run\s+(\d+)', run_input, re.IGNORECASE)
        if not run_match:
            return "ERROR: Run 번호를 찾을 수 없습니다.\n예: 'run 12345' 또는 'run 12345 hv_c=800 hv_s=800'"
        
        run_num = int(run_match.group(1))
        
        # HV 값 파싱 (선택적)
        hv_c_match = re.search(r'hv_c[=:\s]+(\d+)', run_input, re.IGNORECASE)
        hv_s_match = re.search(r'hv_s[=:\s]+(\d+)', run_input, re.IGNORECASE)
        
        # Smart HV Detection: 명시적 입력 vs 이전 AI 제안 사용
        session_id = "default"
        session = _session_manager.get_session(session_id)
        
        center_input = None
        center_match = re.search(r'(?:--center|center)\s*(?:=|:)?\s*([A-Za-z0-9_-]+)', run_input, re.IGNORECASE)
        if center_match:
            center_input = center_match.group(1).strip().upper()
            if session is None:
                _session_manager.start_session(session_id, DEFAULT_TARGET_ADC, DEFAULT_TARGET_ADC)
                session = _session_manager.get_session(session_id)
            if session:
                session["center"] = center_input
        
        if session is None:
            return "ERROR: 세션이 시작되지 않았습니다.\n먼저 'center T1' 명령으로 센터를 등록하거나 target을 설정하세요."
        
        center_value = session.get("center")
        if not center_value:
            return "ERROR: Center 정보가 없습니다. 'center T1' 명령을 먼저 실행해주세요."
        
        if hv_c_match and hv_s_match:
            # 명시적 HV 입력 - 사용자가 직접 조정
            hv_c = int(hv_c_match.group(1))
            hv_s = int(hv_s_match.group(1))
            hv_source = "user_explicit"
        elif session and "last_suggestion" in session:
            # 이전 AI 제안 사용 - 사용자가 AI 제안대로 설정했다고 가정
            hv_c = session["last_suggestion"]["suggested_hv_c"]
            hv_s = session["last_suggestion"]["suggested_hv_s"]
            hv_source = "ai_suggestion"
        else:
            # 첫 번째 run이거나 이전 제안이 없음
            return "ERROR: 첫 번째 run이거나 이전 AI 제안이 없습니다.\nHV 값을 명시해주세요: 'run 12345 hv_c=800 hv_s=800'"
        
        
        # peakADC 분석 실행
        current_c, current_s = run_peak_adc_analysis(run_num, center_value)
        
        if current_c is None or current_s is None:
            return f"ERROR: Run {run_num}에서 ADC 데이터를 가져올 수 없습니다."
        
        # Exponential fitting 병렬 처리
        session_id = "default"
        result = _session_manager.process_parallel_run(
            session_id, run_num, current_c, current_s, hv_c, hv_s, hv_source
        )
        
        return result
        
    except ValueError as e:
        import traceback
        return f"ERROR: 입력값 오류: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
    except Exception as e:
        import traceback
        return f"ERROR: Exponential fitting 분석 실패: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"


@tool
def hveq_add_training_data(data_input: str) -> str:
    """
    새로운 훈련 데이터 추가
    
    Args:
        data_input: "channel=C hv=800 adc=89.3 target=100 change=+8 result=97.1"
    """
    try:
        channel_match = re.search(r'channel[=:\s]+([CS])', data_input, re.IGNORECASE)
        hv_match = re.search(r'hv[=:\s]+(\d+)', data_input, re.IGNORECASE)
        adc_match = re.search(r'adc[=:\s]+([\d.]+)', data_input, re.IGNORECASE)
        target_match = re.search(r'target[=:\s]+([\d.]+)', data_input, re.IGNORECASE)
        change_match = re.search(r'change[=:\s]+([+-]?\d+)', data_input, re.IGNORECASE)
        result_match = re.search(r'result[=:\s]+([\d.]+)', data_input, re.IGNORECASE)
        
        if not all([channel_match, hv_match, adc_match, target_match, change_match, result_match]):
            return "ERROR: 데이터 형식이 잘못되었습니다.\n예: 'channel=C hv=800 adc=89.3 target=100 change=+8 result=97.1'"
        
        channel = channel_match.group(1)
        hv = int(hv_match.group(1))
        adc = float(adc_match.group(1))
        target = float(target_match.group(1))
        change = int(change_match.group(1))
        result_adc = float(result_match.group(1))
        
        result_message = _session_manager.add_measurement_for_learning(
            "default", channel, hv, adc, hv + change, result_adc, target
        )
        
        return result_message
        
    except Exception as e:
        return f"ERROR: 학습 데이터 추가 실패: {str(e)}"

@tool
def hveq_done_channel(done_input: str) -> str:
    """
    채널별 HV equalization 완료 표시
    
    Args:
        done_input: "done" (모든 채널) 또는 "done C" (특정 채널) 또는 "done C S" (여러 채널)
    
    Returns:
        채널 완료 상태 업데이트 결과
    """
    try:
        channels_found = re.findall(r'[CS]', done_input.upper())
        
        if not channels_found:
            channels_to_complete = ["C", "S"]
            completion_type = "모든 채널"
        else:
            channels_to_complete = list(set(channels_found))
            if len(channels_to_complete) == 1:
                completion_type = f"{channels_to_complete[0]} 채널"
            else:
                completion_type = "지정된 채널들"
        
        for channel in channels_to_complete:
            _session_manager.exp_predictor.mark_channel_done(channel)
        
        c_status = "DONE" if not _session_manager.exp_predictor.is_channel_active("C") else "Active"
        s_status = "DONE" if not _session_manager.exp_predictor.is_channel_active("S") else "Active"
        if len(channels_to_complete) == 1:
            result = f"✅ {channels_to_complete[0]} 채널 equalization 완료!"
        elif len(channels_to_complete) == 2:
            result = "✅ 모든 채널 equalization 완료!"
        else:
            result = f"✅ {completion_type} equalization 완료!"
        
        result += f"""

채널 상태:
  C: {c_status}
  S: {s_status}

완료된 채널은 더 이상 HV 조정 suggestion을 제공하지 않습니다."""
        
        if not _session_manager.exp_predictor.is_channel_active("C") and not _session_manager.exp_predictor.is_channel_active("S"):
            reset_message = _session_manager.reset_for_new_pmt("default")
            result += f"\n\n{reset_message}"
            
        return result
        
    except Exception as e:
        return f"ERROR: Done 처리 실패: {str(e)}"

@tool  
def hveq_system_status() -> str:
    """
    Exponential Fitting 시스템 상태 확인
    
    Returns:
        현재 fitting 상태 및 exponential 계수
    """
    try:
        session = _session_manager.get_session("default")
        exp_predictor = _session_manager.exp_predictor
        
        status = f"""🎯 Exponential Fitting System Status

Fitting 상태:
  C 채널: {'Fitted' if exp_predictor.is_fitted['C'] else '❌ 탐색모드'}
  S 채널: {'Fitted' if exp_predictor.is_fitted['S'] else '❌ 탐색모드'}
  Fitting 라이브러리: {'scipy 사용가능' if FITTING_AVAILABLE else '⚠️ 2점 fitting 사용'}

세션 상태:"""
        
        if session:
            status += f"""
  Target: C={session['target_c']}, S={session['target_s']}
  Center: {session.get('center') or '미설정'}
  현재 HV: C={session.get('current_hv_c', 'N/A')}V, S={session.get('current_hv_s', 'N/A')}V
  진행 횟수: {session.get('iteration', 0)}회
  히스토리: {len(session.get('history', []))}개 기록"""
        else:
            status += "\n  ❌ 활성 세션 없음"
        
        status += f"""

Exponential 계수:"""
        
        for channel in ["C", "S"]:
            if exp_predictor.is_fitted[channel]:
                A, B = exp_predictor.coefficients[channel]
                status += f"\n  {channel}: ADC = {A:.2f} * exp({B:.5f} * HV) ({len(exp_predictor.data_points[channel])}점)"
            else:
                status += f"\n  {channel}: 미fitted ({len(exp_predictor.data_points[channel])}점 수집됨)"
        
        return status
        
    except Exception as e:
        return f"ERROR: 상태 확인 실패: {str(e)}"


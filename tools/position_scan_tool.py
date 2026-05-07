#!/usr/bin/env python3
"""
Position Scan Tool for Test Beam Automation
타워 경계 찾기 및 중심 위치 계산
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

try:
    from langchain_core.tools import tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    def tool(func):
        return func
    LANGCHAIN_AVAILABLE = False

try:
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

import sys
from pathlib import Path
tools_dir = Path(__file__).parent
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))
from config_loader import get_data_directory, get_mapping_csv_path, get_path_config




# Data paths (from config_general.yml)
DATA_DIR = get_data_directory()
MAPPING_PATH = get_mapping_csv_path()
RUNNUM_PATH = get_path_config("RunNumberFile")

# Scan 결과 저장 경로
SCAN_RESULTS_DIR = str(Path(get_path_config("PositionResultsFile")).parent)
os.makedirs(SCAN_RESULTS_DIR, exist_ok=True)


# ======================= peakADC 계산 =======================

def count_events_in_file(path: str) -> int:
    """Count total events in a .dat file"""
    filesize = os.path.getsize(path)
    event_size = 64 + 32736*2
    return filesize // event_size


def load_dat_event(path: str, target_ch: int, event_idx: int) -> Optional[np.ndarray]:
    """Load a single event waveform from .dat file"""
    event_size = 64 + 32736*2
    try:
        with open(path, "rb") as f:
            f.seek(event_idx * event_size)
            header = f.read(64)
            data = f.read(32736*2)
        adc = np.frombuffer(data, dtype="<i2")
        reshaped = adc.reshape((1023, 32))
        wf = reshaped[:, target_ch][1:1000]
        return wf
    except Exception:
        return None


def compute_peakADC_100bin(wf: np.ndarray) -> Optional[float]:
    """
    Compute peakADC using first 100 bins for baseline
    
    Args:
        wf: Waveform array
    
    Returns:
        peakADC value (maximum amplitude above baseline)
    """
    if len(wf) < 100:
        return None
    
    baseline = np.mean(wf[:100])
    peakADC = np.max(baseline - wf)
    return float(peakADC)


def smart_valley_cut(values: list, min_valley_height_ratio: float = 0.1, 
                     cut_at_zero: bool = True, bins: int = 200) -> np.ndarray:
    """
    Smart valley cut to remove noise peaks while preserving real signals
    
    Logic:
    - Deep valley (< threshold) → Noise peak → Cut at valley position
    - Shallow valley (≥ threshold) → Double peak (real signals) → Don't cut
    - Cut at first zero count bin from right
    """
    if len(values) == 0:
        return np.array([])
    
    values = np.array(values)
    
    # Create histogram for analysis
    hist, bin_edges = np.histogram(values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Smooth histogram for peak detection
    if SCIPY_AVAILABLE:
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2.0)
    else:
        # Simple moving average fallback
        window = 5
        hist_smooth = np.convolve(hist.astype(float), np.ones(window)/window, mode='same')
    
    # Find peaks (local maxima with significant height)
    peaks = []
    min_peak_height = 0.05 * np.max(hist_smooth)
    
    for i in range(1, len(hist_smooth) - 1):
        if (hist_smooth[i] > hist_smooth[i-1] and 
            hist_smooth[i] > hist_smooth[i+1] and 
            hist_smooth[i] > min_peak_height):
            peaks.append(i)
    
    if len(peaks) < 2:
        filtered_values = values
    else:
        # Find main peak (rightmost significant peak)
        main_peak_idx = peaks[-1]
        main_peak_height = hist_smooth[main_peak_idx]
        
        # Look for valleys to the left of main peak
        best_cut_position = None
        
        for peak_idx in reversed(peaks[:-1]):
            # Find valley between this peak and the main peak
            start_idx = peak_idx
            end_idx = main_peak_idx
            
            if start_idx < end_idx:
                valley_region = hist_smooth[start_idx:end_idx]
                valley_idx_relative = np.argmin(valley_region)
                valley_idx_absolute = start_idx + valley_idx_relative
                valley_height = hist_smooth[valley_idx_absolute]
                
                # Calculate valley height ratio
                valley_height_ratio = valley_height / main_peak_height
                
                # DEEP valley indicates noise peak -> CUT
                if valley_height_ratio < min_valley_height_ratio:
                    best_cut_position = bin_centers[valley_idx_absolute]
                    break
        
        if best_cut_position is None:
            filtered_values = values
        else:
            # Apply vertical cut at the valley position
            filtered_values = values[values >= best_cut_position]
    
    # Apply zero-count cut if requested (right cut from main peak)
    if cut_at_zero and len(filtered_values) > 0:
        hist_for_zero, bin_edges_for_zero = np.histogram(filtered_values, bins=bins)
        bin_centers_for_zero = (bin_edges_for_zero[:-1] + bin_edges_for_zero[1:]) / 2
        
        # Find main peak position in the histogram
        main_peak_bin_idx = np.argmax(hist_for_zero)
        
        # From main peak, go right and find first zero count bin
        zero_cut_position = None
        for i in range(main_peak_bin_idx + 1, len(hist_for_zero)):
            if hist_for_zero[i] == 0:
                zero_cut_position = bin_edges_for_zero[i]
                break
        
        if zero_cut_position is not None:
            filtered_values = filtered_values[filtered_values < zero_cut_position]
    
    return filtered_values


def collect_peakADC_for_run(run_num: int, tower: str) -> list:
    """
    Collect peakADC values from S channel only for a specific run and tower
    
    Args:
        run_num: Run number
        tower: Tower name (e.g., "T4", "T5")
    
    Returns:
        List of peakADC values (S channel only)
    """
    if not os.path.exists(MAPPING_PATH):
        return []
    
    mapping_df = pd.read_csv(MAPPING_PATH)
    peakADC_values = []
    
    # Tower의 S 채널만 사용
    for cs_type in ['S']:
        sub_center = f"{tower}-{cs_type}".strip()
        
        # Find mapping entries
        mapping_rows = mapping_df[mapping_df['pmt'].astype(str).str.strip() == sub_center.strip()]
        
        for _, mrow in mapping_rows.iterrows():
            if pd.isna(mrow['mid']) or pd.isna(mrow['ch']):
                continue
            mid = int(mrow['mid'])
            ch = int(mrow['ch']) - 1
            if not (0 <= ch < 32):
                continue
            
            # Construct file path pattern
            pattern = os.path.join(DATA_DIR, f"Run_{run_num}/Run_{run_num}_Wave/Run_{run_num}_Wave_MID_{mid}/Run_{run_num}_Wave_MID_{mid}_FILE_*.dat")
            target_files = glob.glob(pattern)
            
            if not target_files:
                continue
            
            for target_file in target_files:
                n_events = count_events_in_file(target_file)
                if n_events == 0:
                    continue
                
                for event_idx in range(n_events):
                    wf = load_dat_event(target_file, ch, event_idx)
                    if wf is not None:
                        peakADC = compute_peakADC_100bin(wf)
                        if peakADC is not None:
                            peakADC_values.append(peakADC)
    
    return peakADC_values


def calculate_valley_cut_average(run_num: int, tower: str) -> Tuple[Optional[float], int]:
    """
    Calculate average of peakADC values after valley cut
    
    Args:
        run_num: Run number
        tower: Tower name (e.g., "T4", "T5")
    
    Returns:
        (average, count)
    """
    # Collect peakADC data
    peakADC_data = collect_peakADC_for_run(run_num, tower)
    
    if len(peakADC_data) == 0:
        return None, 0
    
    # Apply valley cut
    filtered_data = smart_valley_cut(peakADC_data, min_valley_height_ratio=0.3, cut_at_zero=True)
    
    if len(filtered_data) == 0:
        return None, 0
    
    # Calculate average
    average = np.mean(filtered_data)
    count = len(filtered_data)
    
    return average, count


# ======================= Position Scan Session =======================

class PositionScanSession:
    """Position scan 세션 관리"""
    
    def __init__(self):
        self.scan_data = {
            "T4": [],    # T4-T5 경계
            "T6": [],    # T5-T6 경계
            "T2": [],    # T2-T5 경계
            "T8": []     # T5-T8 경계
        }
        self.current_direction = None  # "T4", "T6", "T2", "T8"
        self.initial_position = None
        self.scan_active = False
        self.boundaries = {}  # {"T4": x, "T6": x, "T2": y, "T8": y}
    
    def start_scan(self, initial_x: float, initial_y: float, direction: str) -> str:
        """
        Position scan 시작
        
        Args:
            initial_x: 초기 x 좌표
            initial_y: 초기 y 좌표
            direction: 타워 이름 ("T4", "T6", "T2", "T8")
        
        Returns:
            시작 확인 메시지
        """
        self.initial_position = {"x": initial_x, "y": initial_y}
        self.current_direction = direction
        self.scan_active = True
        
        return f"""✅ Position scan 시작!

타워: {direction}
초기 위치: x={initial_x:.2f}mm, y={initial_y:.2f}mm

이제 run을 실행하고 add_scan_point로 데이터를 추가하세요."""
    
    def add_scan_point(self, run_num: int, x: float, y: float, 
                       tower1: str, tower2: str) -> str:
        """
        Scan 데이터 포인트 추가
        
        Args:
            run_num: Run 번호
            x: 현재 x 좌표
            y: 현재 y 좌표
            tower1: 첫 번째 타워 (예: "T4")
            tower2: 두 번째 타워 (예: "T5")
        
        Returns:
            추가 결과 메시지
        """
        # scan_active가 아니면 자동 초기화 (PS_data_gen.py 워크플로우에 맞춤)
        if not self.scan_active:
            # tower1, tower2로 direction 자동 추론 (타워 이름으로)
            if tower1 == "T4" and tower2 == "T5":
                direction = "T4"
            elif tower1 == "T5" and tower2 == "T6":
                direction = "T6"
            elif tower1 == "T2" and tower2 == "T5":
                direction = "T2"
            elif tower1 == "T5" and tower2 == "T8":
                direction = "T8"
            else:
                return f"ERROR: 알 수 없는 타워 조합 ({tower1}, {tower2}). 지원되는 조합: (T4,T5), (T5,T6), (T2,T5), (T5,T8)"
            
            # 자동 초기화
            self.initial_position = {"x": x, "y": y}
            self.current_direction = direction
            self.scan_active = True
        
        # peakADC 계산
        avg1, count1 = calculate_valley_cut_average(run_num, tower1)
        avg2, count2 = calculate_valley_cut_average(run_num, tower2)
        
        if avg1 is None or avg2 is None:
            return f"ERROR: Run {run_num}에서 peakADC 데이터를 가져올 수 없습니다."
        
        # 데이터 저장
        data_point = {
            "run": run_num,
            "x": x,
            "y": y,
            f"{tower1}_peakADC": avg1,
            f"{tower2}_peakADC": avg2
        }
        
        self.scan_data[self.current_direction].append(data_point)
        
        # 교차 확인
        crossed = self._check_crossing()
        
        result = f"""Run {run_num} 데이터 추가:

위치: x={x:.2f}mm, y={y:.2f}mm
{tower1} peakADC: {avg1:.1f} (이벤트 수: {count1})
{tower2} peakADC: {avg2:.1f} (이벤트 수: {count2})

총 {len(self.scan_data[self.current_direction])}개 포인트 수집됨"""
        
        if crossed:
            result += f"\n\n⚠️ 교차점 감지! 경계를 찾았을 가능성이 높습니다."
        
        return result
    
    def _check_crossing(self) -> bool:
        """신호 교차 확인 (tower1 < tower2 → tower1 > tower2)"""
        data = self.scan_data[self.current_direction]
        if len(data) < 2:
            return False
        
        # 마지막 두 포인트 비교
        prev_point = data[-2]
        curr_point = data[-1]
        
        # 키 이름 찾기
        tower1_key = [k for k in prev_point.keys() if k.endswith("_peakADC")][0]
        tower2_key = [k for k in prev_point.keys() if k.endswith("_peakADC")][1]
        
        prev_t1 = prev_point[tower1_key]
        prev_t2 = prev_point[tower2_key]
        curr_t1 = curr_point[tower1_key]
        curr_t2 = curr_point[tower2_key]
        
        # 교차 확인
        if (prev_t1 < prev_t2) and (curr_t1 > curr_t2):
            return True
        if (prev_t1 > prev_t2) and (curr_t1 < curr_t2):
            return True
        
        return False
    
    def find_boundary(self) -> Optional[float]:
        """
        현재 방향의 경계 좌표 계산 (선형 보간)
        
        Returns:
            경계 좌표 (x 또는 y)
        """
        data = self.scan_data[self.current_direction]
        if len(data) < 2:
            return None
        
        # 교차점 찾기
        for i in range(len(data) - 1):
            curr = data[i]
            next = data[i + 1]
            
            # 키 이름
            keys = [k for k in curr.keys() if k.endswith("_peakADC")]
            tower1_key = keys[0]
            tower2_key = keys[1]
            
            curr_t1 = curr[tower1_key]
            curr_t2 = curr[tower2_key]
            next_t1 = next[tower1_key]
            next_t2 = next[tower2_key]
            
            # 교차 확인
            if (curr_t1 < curr_t2 and next_t1 > next_t2) or \
               (curr_t1 > curr_t2 and next_t1 < next_t2):
                
                # 선형 보간으로 교차점 계산 (Excel 방식과 동일)
                # tower1(x) = tower2(x)인 지점을 찾음
                # tower1과 tower2를 각각 선형 보간한 후 교차점 계산
                if self.current_direction in ["T4", "T6"]:
                    # x 좌표 보간
                    x_curr = curr["x"]
                    x_next = next["x"]
                    
                    # tower1과 tower2의 선형 회귀 계수
                    m1 = (next_t1 - curr_t1) / (x_next - x_curr)  # tower1의 기울기
                    b1 = curr_t1 - m1 * x_curr                      # tower1의 y절편
                    m2 = (next_t2 - curr_t2) / (x_next - x_curr)  # tower2의 기울기
                    b2 = curr_t2 - m2 * x_curr                      # tower2의 y절편
                    
                    # tower1(x) = tower2(x)인 지점: m1*x + b1 = m2*x + b2
                    # (m1 - m2)*x = b2 - b1
                    if abs(m1 - m2) > 1e-10:  # 기울기가 다를 때만
                        x_boundary = (b2 - b1) / (m1 - m2)
                        return x_boundary
                else:  # T2, T8
                    # y 좌표 보간
                    y_curr = curr["y"]
                    y_next = next["y"]
                    
                    # tower1과 tower2의 선형 회귀 계수
                    m1 = (next_t1 - curr_t1) / (y_next - y_curr)  # tower1의 기울기
                    b1 = curr_t1 - m1 * y_curr                      # tower1의 x절편
                    m2 = (next_t2 - curr_t2) / (y_next - y_curr)  # tower2의 기울기
                    b2 = curr_t2 - m2 * y_curr                      # tower2의 x절편
                    
                    # tower1(y) = tower2(y)인 지점: m1*y + b1 = m2*y + b2
                    if abs(m1 - m2) > 1e-10:  # 기울기가 다를 때만
                        y_boundary = (b2 - b1) / (m1 - m2)
                        return y_boundary
        
        return None
    
    def save_direction_csv(self, direction: str, output_dir: Optional[str] = None) -> Optional[str]:
        """
        특정 타워의 스캔 데이터를 CSV로 저장
        
        Args:
            direction: 타워 이름 ("T4", "T6", "T2", "T8")
            output_dir: 저장 경로 (None이면 기본 경로)
        
        Returns:
            CSV 파일 경로 또는 None
        """
        if direction not in self.scan_data or len(self.scan_data[direction]) == 0:
            return None
        
        if output_dir is None:
            output_dir = SCAN_RESULTS_DIR
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame(self.scan_data[direction])
        csv_path = os.path.join(output_dir, f"position_scan_{direction}_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    def complete_direction(self) -> str:
        """
        현재 방향 스캔 완료 및 결과 반환 (JSON 형식)
        
        Returns:
            JSON 형식의 결과:
            {
                "status": "success",
                "direction": "left",
                "boundary": 985.0,
                "csv_path": "...",
                "data_points": 5,
                "message": "..."
            }
        """
        boundary = self.find_boundary()
        
        if boundary is None:
            return json.dumps({
                "status": "error",
                "message": "경계를 찾을 수 없습니다. 더 많은 데이터가 필요합니다."
            }, ensure_ascii=False, indent=2)
        
        # 경계 저장
        self.boundaries[self.current_direction] = boundary
        
        # CSV 저장
        csv_path = self.save_direction_csv(self.current_direction)
        
        coord_type = "x" if self.current_direction in ["T4", "T6"] else "y"
        data_points = len(self.scan_data[self.current_direction])
        
        # direction에서 tower1, tower2 역추론
        if self.current_direction == "T4":
            tower1, tower2 = "T4", "T5"
        elif self.current_direction == "T6":
            tower1, tower2 = "T5", "T6"
        elif self.current_direction == "T2":
            tower1, tower2 = "T2", "T5"
        elif self.current_direction == "T8":
            tower1, tower2 = "T5", "T8"
        else:
            tower1, tower2 = "T5", "T5"  # fallback
        
        # PS_data_gen.py 형식에 맞춘 반환값
        if coord_type == "x":
            boundary_x = boundary
            boundary_y = self.initial_position["y"]
        else:
            boundary_x = self.initial_position["x"]
            boundary_y = boundary
        
        result = {
            "status": "success",
            "message": f"{tower1}와 {tower2}의 경계는 x={boundary_x:.0f}mm, y={boundary_y:.0f}mm 입니다.",
            "csv_path": csv_path if csv_path else f"/path/to/position_scan_{self.current_direction}_*.csv",
            "boundary": boundary
        }
        
        self.scan_active = False
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    def save_results(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        완료된 모든 방향의 스캔 데이터를 CSV로 저장
        
        Args:
            output_dir: 저장 경로 (None이면 기본 경로)
        
        Returns:
            {"csv_files": {direction: csv_path, ...}}
        """
        if output_dir is None:
            output_dir = SCAN_RESULTS_DIR
        
        csv_paths = {}
        for direction in ["T4", "T6", "T2", "T8"]:
            if direction in self.scan_data and len(self.scan_data[direction]) > 0:
                csv_path = self.save_direction_csv(direction, output_dir)
                if csv_path:
                    csv_paths[direction] = csv_path
        
        return {
            "csv_files": csv_paths
        }
    
    def reset(self):
        """세션 초기화"""
        self.scan_data = {"T4": [], "T6": [], "T2": [], "T8": []}
        self.current_direction = None
        self.initial_position = None
        self.scan_active = False
        self.boundaries = {}


# ======================= Global Session =======================
_scan_session = PositionScanSession()


# ======================= LangChain Tools =======================

@tool
def get_peakadc_average(tower: str, run_number: Optional[int] = None) -> str:
    """
    특정 타워의 peakADC 평균값을 계산합니다 (valley cut 적용).
    
    Args:
        tower: 타워 이름 (예: "T4", "T5")
        run_number: Run 번호 (None이면 runnum.txt에서 자동 읽기)
    
    Returns:
        peakADC 평균값과 이벤트 개수
    """
    try:
        # Run number 결정
        if run_number is None:
            try:
                with open(RUNNUM_PATH, 'r') as f:
                    # Run이 종료된 후 runnum.txt가 다음 번호로 업데이트되므로, 
                    # 방금 종료된 Run 분석을 위해 -1을 수행함
                    val = f.read().strip()
                    run_number = int(val) - 1
            except Exception:
                return "ERROR: Run number를 가져올 수 없습니다."
        
        avg, count = calculate_valley_cut_average(run_number, tower.upper())
        
        if avg is None:
            return f"ERROR: Run {run_number}에서 {tower} peakADC 데이터를 가져올 수 없습니다."
        
        return f"""Run {run_number} - {tower} peakADC Analysis:

평균 peakADC: {avg:.2f}
이벤트 수: {count}"""
        
    except Exception as e:
        return f"ERROR: peakADC 계산 실패: {str(e)}"


@tool
def position_scan_start(initial_x: float, initial_y: float, direction: str) -> str:
    """
    Position scan을 시작합니다.
    
    Args:
        initial_x: 초기 x 좌표 (mm)
        initial_y: 초기 y 좌표 (mm)
        direction: 타워 이름 ("T4", "T6", "T2", "T8")
    
    Returns:
        시작 확인 메시지
    """
    try:
        result = _scan_session.start_scan(initial_x, initial_y, direction)
        return result
    except Exception as e:
        return f"ERROR: Position scan 시작 실패: {str(e)}"


@tool
def position_scan_add_point(run_number: int, x: float, y: float, 
                           tower1: str, tower2: str) -> str:
    """
    Position scan 데이터 포인트를 추가합니다.
    
    Args:
        run_number: Run 번호
        x: 현재 x 좌표 (mm)
        y: 현재 y 좌표 (mm)
        tower1: 첫 번째 타워 (예: "T4")
        tower2: 두 번째 타워 (예: "T5")
    
    Returns:
        데이터 추가 결과 및 교차점 감지 여부
    """
    try:
        result = _scan_session.add_scan_point(run_number, x, y, tower1, tower2)
        return result
    except Exception as e:
        return f"ERROR: 데이터 포인트 추가 실패: {str(e)}"


@tool
def position_scan_complete_direction() -> str:
    """
    현재 방향의 position scan을 완료하고 경계를 계산합니다.
    
    Returns:
        경계 계산 결과
    """
    try:
        result = _scan_session.complete_direction()
        return result
    except Exception as e:
        return f"ERROR: 방향 완료 실패: {str(e)}"


@tool
def position_scan_calculate_center() -> str:
    """
    ⚠️ DEPRECATED: 이 함수는 더 이상 사용되지 않습니다.
    
    Agent가 4개 방향(left, right, up, down)의 boundary 값을 수집한 후,
    다음 공식으로 직접 계산하세요:
    
    offset_x = (left + right) / 2
    offset_y = (up + down) / 2
    tower_width = right - left
    tower_height = up - down
    
    각 방향의 boundary 값은 position_scan_complete_direction()의 반환값에서 얻을 수 있습니다.
    """
    return json.dumps({
        "status": "deprecated",
        "message": """⚠️ 이 함수는 더 이상 사용되지 않습니다.

Agent가 각 방향의 boundary 값을 수집한 후 직접 계산하세요:

1. position_scan_complete_direction()을 4번 호출하여 각 방향의 boundary 수집
2. 수집한 boundary 값으로 계산:
   - offset_x = (left + right) / 2
   - offset_y = (up + down) / 2
   - tower_width = right - left
   - tower_height = up - down
3. 계산 결과를 JSON으로 저장하여 position_calculator_tool에 전달"""
    }, ensure_ascii=False, indent=2)


@tool
def position_scan_save() -> str:
    """
    완료된 모든 방향의 스캔 데이터를 CSV 파일로 저장합니다.
    
    Returns:
        저장된 CSV 파일 경로 목록
    """
    try:
        paths = _scan_session.save_results()
        
        result_lines = ["✅ Position scan CSV 파일 저장 완료!", ""]
        result_lines.append("저장된 CSV 파일:")
        if len(paths["csv_files"]) == 0:
            result_lines.append("  저장할 데이터가 없습니다.")
        else:
            for direction, path in paths["csv_files"].items():
                result_lines.append(f"  {direction}: {path}")
        
        result_lines.append("")
        result_lines.append("💡 각 방향의 boundary 값을 수집한 후, Agent가 직접 계산하여")
        result_lines.append("   JSON으로 저장하고 position_calculator_tool에 전달하세요.")
        
        return "\n".join(result_lines)
        
    except Exception as e:
        return f"ERROR: 결과 저장 실패: {str(e)}"




@tool
def position_scan_status() -> str:
    """
    Position scan의 현재 상태를 확인합니다.
    
    Returns:
        현재 상태 정보 (JSON 형식)
    """
    try:
        status = {
            "active": _scan_session.scan_active,
            "current_direction": _scan_session.current_direction,
            "data_points": {
                direction: len(data) 
                for direction, data in _scan_session.scan_data.items()
            },
            "boundaries": _scan_session.boundaries.copy()
        }
        
        # 사람이 읽기 좋은 메시지 생성
        status_lines = ["📍 Position Scan 상태", ""]
        status_lines.append(f"활성: {'✅' if status['active'] else '❌'}")
        status_lines.append(f"현재 방향: {status['current_direction'] or 'N/A'}")
        status_lines.append("")
        
        status_lines.append("수집된 데이터:")
        for direction, count in status['data_points'].items():
            status_lines.append(f"  {direction}: {count}개 포인트")
        
        status_lines.append("")
        status_lines.append("완료된 경계:")
        if len(status['boundaries']) == 0:
            status_lines.append("  없음")
        else:
            for direction, value in status['boundaries'].items():
                coord_type = "x" if direction in ["T4", "T6"] else "y"
                status_lines.append(f"  {direction}: {coord_type}={value:.2f}mm")
        
        status_lines.append("")
        if len(status['boundaries']) < 4:
            remaining = 4 - len(status['boundaries'])
            status_lines.append(f"💡 {remaining}개 타워가 더 필요합니다.")
            status_lines.append("   모든 타워 완료 후, Agent가 boundary 값으로 계산하세요:")
            status_lines.append("   - offset_x = (T4 + T6) / 2")
            status_lines.append("   - offset_y = (T2 + T8) / 2")
            status_lines.append("   - tower_width = T6 - T4")
            status_lines.append("   - tower_height = T2 - T8")
        
        status["message"] = "\n".join(status_lines)
        return json.dumps(status, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"상태 확인 실패: {str(e)}"
        }, ensure_ascii=False, indent=2)


@tool
def position_scan_reset() -> str:
    """
    Position scan 세션을 초기화합니다.
    
    Returns:
        초기화 확인 메시지
    """
    try:
        _scan_session.reset()
        return "✅ Position scan 세션 초기화 완료."
    except Exception as e:
        return f"ERROR: 초기화 실패: {str(e)}"

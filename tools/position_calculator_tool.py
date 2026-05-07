#!/usr/bin/env python3
"""
Position Calculator Tool for Test Beam Automation
타워 위치 계산 도구 (T5 기준 상대 위치 계산)
"""

import json
import math
from typing import Dict, Any, Optional, Tuple
from .config_loader import get_path_config

# ===== 경로 설정 (Config from YAML) =====
POSITION_RESULTS_FILE = get_path_config("PositionResultsFile")

try:
    from langchain_core.tools import tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    def tool(func):
        return func
    LANGCHAIN_AVAILABLE = False


# ======================= 상수 정의 =======================
# Tilting Axis 거리 (mm)
# 모듈을 들 때 드는 점부터 축까지의 거리
# 스프레드시트 수식에서 사용되는 53.5 값
TILTING_AXIS = 53.5  # mm

# Rotation Axis 상수
# Rotation Axis 기준 각도 (degree)
# 스프레드시트 수식에서 사용되는 11.72 값
ROTATION_AXIS_ANGLE = 11.72  # degree

# Rotation Axis 거리 (mm)
# 스프레드시트 수식에서 사용되는 2.4 값
ROTATION_AXIS_DIST = 2.4  # mm


# ======================= Tower Layout =======================
"""
타워 레이아웃 (3x3 그리드):

    T1  T2  T3
    T4  T5  T6
    T7  T8  T9

T5가 중심이며, 다른 타워들은 SWITCH 함수를 통해 오프셋 계산됨.
"""

# ======================= Position Calculator =======================

class PositionCalculator:
    """타워 위치 계산기 (스프레드시트 수식 구조 반영)"""
    
    def __init__(self):
        # 타워 간격 상수 (스프레드시트의 B41, C41)
        self.tower_width = None   # B41: X 방향 타워 간격 = Tower width (mm)
        self.tower_height = None  # C41: Y 방향 타워 간격 = Tower height (mm)
        
        # T5 중심 좌표 (Offset X, Y)
        self.offset_x = None  # T5 중심 x 좌표 (mm) - B45, C45의 기준점
        self.offset_y = None  # T5 중심 y 좌표 (mm) - B45, C45의 기준점
        
        # Rotation, Tilting 각도 (degree)
        self.rotation = 0.0  # Rotation 각도 (degree)
        self.tilting = 0.0   # Tilting 각도 (degree)
        
        self.scan_completed = False
    
    def set_position_scan_result(self, offset_x: float, offset_y: float,
                                tower_width: float, tower_height: float) -> str:
        """
        Position scan 결과 저장
        
        Args:
            offset_x: T5 중심 x 좌표 (mm) - Offset X
            offset_y: T5 중심 y 좌표 (mm) - Offset Y
            tower_width: X 방향 타워 간격 = Tower width (mm) - 스프레드시트 B41
            tower_height: Y 방향 타워 간격 = Tower height (mm) - 스프레드시트 C41
        
        Returns:
            저장 확인 메시지
        """
        self.offset_x = float(offset_x)
        self.offset_y = float(offset_y)
        self.tower_width = float(tower_width)
        self.tower_height = float(tower_height)
        
        self.scan_completed = True
        
        return f"""✅ Position scan 결과 저장 완료!

Offset X (T5 중심 x): {offset_x:.2f}mm
Offset Y (T5 중심 y): {offset_y:.2f}mm
Tower Width ={tower_width:.2f}mm
Tower Height={tower_height:.2f}mm

이제 모든 타워의 위치를 계산할 수 있습니다."""
    
    def _calculate_tower_offset_x(self, tower: str) -> float:
        """
        타워별 X 방향 오프셋 계산 (B43 수식)
        
        SWITCH(B6, 
          "T1", B41, "T4", B41, "T7", B41,
          "T2", 0,   "T5", 0,   "T8", 0,
          "T3", -B41, "T6", -B41, "T9", -B41)
        
        Args:
            tower: 타워 이름
        
        Returns:
            X 방향 오프셋
        """
        if tower == "T1" or tower == "T4" or tower == "T7":
            return self.tower_width
        elif tower == "T2" or tower == "T5" or tower == "T8":
            return 0.0
        elif tower == "T3" or tower == "T6" or tower == "T9":
            return -self.tower_width
        else:
            return 0.0
    
    def _calculate_tower_offset_y(self, tower: str) -> float:
        """
        타워별 Y 방향 오프셋 계산 (C43 수식)
        
        SWITCH(B6,
          "T1", -C41, "T2", -C41, "T3", -C41,
          "T4", 0,    "T5", 0,    "T6", 0,
          "T7", C41,  "T8", C41,  "T9", C41)
        
        Args:
            tower: 타워 이름
        
        Returns:
            Y 방향 오프셋
        """
        if tower == "T1" or tower == "T2" or tower == "T3":
            return -self.tower_height
        elif tower == "T4" or tower == "T5" or tower == "T6":
            return 0.0
        elif tower == "T7" or tower == "T8" or tower == "T9":
            return self.tower_height
        else:
            return 0.0
    
    def calculate_tower_position(self, tower: str, 
                                rotation: Optional[float] = None,
                                tilting: Optional[float] = None) -> Optional[Dict[str, float]]:
        """
        특정 타워의 중심 위치 계산 (Rotation/Tilting 적용)
        
        계산 구조:
        - B43 = 타워별 X 오프셋 (SWITCH 함수)
        - C43 = 타워별 Y 오프셋 (SWITCH 함수)
        - B44 = 0 (P5 Center 고정)
        - C44 = 0 (P5 Center 고정)
        - B45 = T5 중심 X + B43 + B44 = T5 중심 X + B43
        - C45 = T5 중심 Y + C43 + C44 = T5 중심 Y + C43
        - C46 = C45 - LIFT_POINT_TO_AXIS_DISTANCE * sin(RADIANS(tilting))
        
        Args:
            tower: 타워 이름 (예: "T1", "T5", "T9")
            rotation: Rotation 각도 (degree, None이면 self.rotation 사용)
            tilting: Tilting 각도 (degree, None이면 self.tilting 사용)
        
        Returns:
            {"x": float, "y": float} 또는 None (scan 결과 없으면)
        """
        if not self.scan_completed:
            return None
        
        tower = tower.upper()
        
        # 유효한 타워인지 확인 (T1-T9)
        valid_towers = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]
        if tower not in valid_towers:
            return None
        
        # Rotation, Tilting 각도 결정
        if rotation is None:
            rotation = self.rotation
        if tilting is None:
            tilting = self.tilting
        
        # B43, C43: 타워별 오프셋 계산
        offset_x = self._calculate_tower_offset_x(tower)  # B43
        offset_y = self._calculate_tower_offset_y(tower)  # C43
        
        # B44, C44: Position 오프셋 (P5 Center 고정이므로 0)
        position_offset_x = 0.0  # B44
        position_offset_y = 0.0  # C44
        
        # B45, C45: 기본 좌표 계산
        # B45 = B43 (타워별 X 오프셋만)
        # C45 = C43 (타워별 Y 오프셋만)
        b45 = offset_x + position_offset_x  # B45 = B43 + B44 (B44=0이므로 B45 = B43)
        c45 = offset_y + position_offset_y  # C45 = C43 + C44 (C44=0이므로 C45 = C43)
        
        # Rotation 적용 (B46 수식)
        # B46 = B45 + F6 + (ROTATION_AXIS_DIST * sin(RADIANS(ROTATION_AXIS_ANGLE) + RADIANS(rotation)) 
        #                   - ROTATION_AXIS_DIST * sin(RADIANS(ROTATION_AXIS_ANGLE)))
        # B45 = B43 (타워별 X 오프셋)
        # F6 = T5 중심 X 값 (offset_x)
        if rotation != 0.0:
            rotation_rad = math.radians(rotation)
            base_angle_rad = math.radians(ROTATION_AXIS_ANGLE)
            rotation_term = (ROTATION_AXIS_DIST * math.sin(base_angle_rad + rotation_rad) 
                           - ROTATION_AXIS_DIST * math.sin(base_angle_rad))
            x = b45 + self.offset_x + rotation_term  # B46 = B45 + F6 + rotation_term
        else:
            x = b45 + self.offset_x  # B45 + F6
        
        # Tilting 적용 (C46 수식)
        # C46 = C45 + G6 - TILTING_AXIS * sin(RADIANS(tilting))
        # C45 = C43 (타워별 Y 오프셋)
        # G6 = T5 중심 Y 값 (offset_y)
        if tilting != 0.0:
            tilting_rad = math.radians(tilting)
            y = c45 + self.offset_y - TILTING_AXIS * math.sin(tilting_rad)  # C46 = C45 + G6 - tilting_term
        else:
            y = c45 + self.offset_y  # C45 + G6
        
        return {"x": x, "y": y}
    
    def calculate_all_positions(self, rotation: Optional[float] = None,
                                tilting: Optional[float] = None) -> Optional[Dict[str, Dict[str, float]]]:
        """
        모든 타워의 위치 계산
        
        Args:
            rotation: Rotation 각도 (degree, None이면 self.rotation 사용)
            tilting: Tilting 각도 (degree, None이면 self.tilting 사용)
        
        Returns:
            {"T1": {"x": float, "y": float}, "T2": ..., ...}
        """
        if not self.scan_completed:
            return None
        
        positions = {}
        # 모든 타워 (T1-T9) 위치 계산
        for tower in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]:
            positions[tower] = self.calculate_tower_position(tower, rotation, tilting)
        
        return positions
    
    def get_status(self) -> Dict[str, Any]:
        """
        현재 상태 확인
        
        Returns:
            상태 정보 딕셔너리
        """
        if not self.scan_completed:
            return {
                "scan_completed": False,
                "message": "Position scan이 아직 완료되지 않았습니다."
            }
        
        return {
            "scan_completed": True,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "tower_spacing": {
                "x": self.tower_width,   # B41: Tower width
                "y": self.tower_height   # C41: Tower height
            },
            "all_positions": self.calculate_all_positions()
        }
    
    def reset(self):
        """데이터 초기화"""
        self.offset_x = None
        self.offset_y = None
        self.tower_width = None
        self.tower_height = None
        self.rotation = 0.0
        self.tilting = 0.0
        self.scan_completed = False


# ======================= Global Calculator =======================
_position_calculator = PositionCalculator()

# 자동 데이터 로드 시도
def _auto_load_results():
    import os
    if os.path.exists(POSITION_RESULTS_FILE):
        try:
            with open(POSITION_RESULTS_FILE, 'r') as f:
                data = json.load(f)
            _position_calculator.set_position_scan_result(
                data["offset_x"], data["offset_y"],
                data["tower_width"], data["tower_height"]
            )
            # 초기 Rotation/Tilting 설정 (EnergyScanAgent 요구사항에 맞춤)
            _position_calculator.rotation = 1.5
            _position_calculator.tilting = 1.0
            print(f"✅ Position Calculator: 자동으로 스캔 결과를 로드했습니다. ({POSITION_RESULTS_FILE})")
        except Exception as e:
            print(f"⚠️ Position Calculator: 자동 로드 실패: {e}")

_auto_load_results()


# ======================= LangChain Tools =======================

@tool
def position_scan_save(offset_x: float, offset_y: float, 
                      tower_width: float, tower_height: float) -> str:
    """
    Position scan 결과를 저장합니다.
    
    Args:
        offset_x: T5 타워 중심 x 좌표 (mm) - Offset X
        offset_y: T5 타워 중심 y 좌표 (mm) - Offset Y
        tower_width: X 방향 타워 간격 = Tower width (mm) - 스프레드시트 B41
        tower_height: Y 방향 타워 간격 = Tower height (mm) - 스프레드시트 C41
    
    Returns:
        저장 확인 메시지
    """
    try:
        result = _position_calculator.set_position_scan_result(
            float(offset_x), float(offset_y), 
            float(tower_width), float(tower_height)
        )
        return result
    except Exception as e:
        return f"ERROR: Position scan 결과 저장 실패: {str(e)}"


@tool
def position_scan_save_from_json(json_file_path: str) -> str:
    """
    Position scan 결과 JSON 파일을 읽어서 자동으로 저장합니다.
    
    Args:
        json_file_path: position_scan_tool의 position_scan_save()로 저장된 JSON 파일 경로
    
    Returns:
        저장 결과 메시지
    """
    try:
        import os
        if not os.path.exists(json_file_path):
            return f"ERROR: JSON 파일을 찾을 수 없습니다: {json_file_path}"
        
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # JSON에서 필요한 값 추출
        if "offset_x" not in data or "offset_y" not in data:
            return "ERROR: JSON 파일에 offset_x 또는 offset_y 정보가 없습니다."
        
        offset_x = data["offset_x"]
        offset_y = data["offset_y"]
        
        if "tower_width" not in data or "tower_height" not in data:
            return "ERROR: JSON 파일에 tower_width 또는 tower_height 정보가 없습니다."
        
        tower_width = data["tower_width"]
        tower_height = data["tower_height"]
        
        # 저장
        return position_scan_save(offset_x, offset_y, tower_width, tower_height)
        
    except json.JSONDecodeError as e:
        return f"ERROR: JSON 파일 파싱 실패: {str(e)}"
    except Exception as e:
        return f"ERROR: 저장 실패: {str(e)}"


@tool
def get_tower_position(tower: str, rotation: Optional[float] = None, 
                      tilting: Optional[float] = None) -> str:
    """
    특정 타워의 중심 위치를 계산합니다 (Rotation/Tilting 적용).
    
    Args:
        tower: 타워 이름 (예: "T1", "T4", "T5", "T8", "T9" 등 T1-T9 중 하나)
        rotation: Rotation 각도 (degree, 선택적, None이면 저장된 값 사용)
        tilting: Tilting 각도 (degree, 선택적, None이면 저장된 값 사용)
    
    Returns:
        타워 위치 정보 (JSON 형식)
        {
            "status": "success",
            "tower": "T5",
            "position": {"x": 10.5, "y": -2.3},
            "rotation": 1.5,
            "tilting": 1.0,
            "message": "T5 타워 위치: x=10.50mm, y=-2.30mm"
        }
    """
    try:
        tower = tower.upper()
        
        # 유효한 타워인지 확인 (T1-T9)
        valid_towers = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]
        if tower not in valid_towers:
            return json.dumps({
                "status": "error",
                "message": f"알 수 없는 타워: {tower}. T1-T9 중 하나를 입력하세요. (예: T4, T5, T8)"
            }, ensure_ascii=False, indent=2)
        
        if not _position_calculator.scan_completed:
            return json.dumps({
                "status": "error",
                "message": "Position scan이 완료되지 않았습니다. 먼저 position_scan_save 또는 position_scan_save_from_json을 호출하세요."
            }, ensure_ascii=False, indent=2)
        
        position = _position_calculator.calculate_tower_position(tower, rotation, tilting)
        
        if position is None:
            return json.dumps({
                "status": "error",
                "message": f"위치 계산 실패: {tower}"
            }, ensure_ascii=False, indent=2)
        
        result = {
            "status": "success",
            "tower": tower,
            "position": position,
            "rotation": rotation if rotation is not None else _position_calculator.rotation,
            "tilting": tilting if tilting is not None else _position_calculator.tilting,
            "message": f"{tower} 타워 위치: x={position['x']:.2f}mm, y={position['y']:.2f}mm"
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"위치 계산 실패: {str(e)}"
        }, ensure_ascii=False, indent=2)


@tool
def set_rotation_tilting(rotation: float, tilting: float) -> str:
    """
    Rotation과 Tilting 각도를 설정합니다.
    
    Args:
        rotation: Rotation 각도 (degree)
        tilting: Tilting 각도 (degree)
    
    Returns:
        설정 확인 메시지
    """
    try:
        _position_calculator.rotation = float(rotation)
        _position_calculator.tilting = float(tilting)
        return f"""✅ Rotation/Tilting 설정 완료!

Rotation: {rotation:.2f}°
Tilting: {tilting:.2f}°

이제 get_tower_position으로 위치를 계산하면 Rotation/Tilting이 적용됩니다."""
    except Exception as e:
        return f"ERROR: Rotation/Tilting 설정 실패: {str(e)}"


@tool
def get_all_tower_positions(rotation: Optional[float] = None,
                            tilting: Optional[float] = None) -> str:
    """
    모든 타워의 위치를 계산합니다 (Rotation/Tilting 적용).
    
    Args:
        rotation: Rotation 각도 (degree, 선택적, None이면 저장된 값 사용)
        tilting: Tilting 각도 (degree, 선택적, None이면 저장된 값 사용)
    
    Returns:
        모든 타워의 위치 정보 (JSON 형식)
        {
            "status": "success",
            "positions": {
                "T1": {"x": 5.5, "y": 2.7},
                "T2": {"x": 10.5, "y": 2.7},
                ...
            }
        }
    """
    try:
        if not _position_calculator.scan_completed:
            return json.dumps({
                "status": "error",
                "message": "Position scan이 완료되지 않았습니다. 먼저 position_scan_save를 호출하세요."
            }, ensure_ascii=False, indent=2)
        
        positions = _position_calculator.calculate_all_positions(rotation, tilting)
        
        # 사람이 읽기 좋은 메시지 생성
        message_lines = ["모든 타워 위치:"]
        for row in ["123", "456", "789"]:
            row_positions = []
            for i in row:
                tower = f"T{i}"
                pos = positions[tower]
                row_positions.append(f"{tower}({pos['x']:.1f},{pos['y']:.1f})")
            message_lines.append("  " + "  ".join(row_positions))
        
        result = {
            "status": "success",
            "offset_x": _position_calculator.offset_x,
            "offset_y": _position_calculator.offset_y,
            "tower_spacing": {
                "x": _position_calculator.tower_width,   # B41: Tower width
                "y": _position_calculator.tower_height   # C41: Tower height
            },
            "positions": positions,
            "message": "\n".join(message_lines)
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"위치 계산 실패: {str(e)}"
        }, ensure_ascii=False, indent=2)


@tool
def position_calculator_status() -> str:
    """
    Position calculator의 현재 상태를 확인합니다.
    
    Returns:
        상태 정보 (JSON 형식)
    """
    try:
        status = _position_calculator.get_status()
        
        if not status["scan_completed"]:
            return json.dumps(status, ensure_ascii=False, indent=2)
        
        # 메시지 추가
        offset_x = status["offset_x"]
        offset_y = status["offset_y"]
        spacing = status["tower_spacing"]
        status["message"] = f"""📍 Position Calculator 상태

Position Scan: ✅ 완료
Offset X (T5 중심 x): {offset_x:.2f}mm
Offset Y (T5 중심 y): {offset_y:.2f}mm
타워 간격 (B41, C41): X={spacing['x']:.2f}mm (Tower width), Y={spacing['y']:.2f}mm (Tower height)

모든 타워 위치 계산 가능 (Rotation/Tilting 미적용)"""
        
        return json.dumps(status, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"상태 확인 실패: {str(e)}"
        }, ensure_ascii=False, indent=2)


@tool
def position_calculator_reset() -> str:
    """
    Position calculator를 초기화합니다.
    
    Returns:
        초기화 확인 메시지
    """
    try:
        _position_calculator.reset()
        return "✅ Position calculator 초기화 완료. 새로운 position scan을 진행하세요."
    except Exception as e:
        return f"ERROR: 초기화 실패: {str(e)}"


# ======================= Direct Access Functions =======================

def calculate_position(tower: str) -> Optional[Dict[str, float]]:
    """
    직접 접근용 함수 (tool decorator 없이)
    
    Args:
        tower: 타워 이름 (예: "T1", "T5")
    
    Returns:
        {"x": float, "y": float} 또는 None
    """
    return _position_calculator.calculate_tower_position(tower)


def is_scan_completed() -> bool:
    """Position scan 완료 여부 확인"""
    return _position_calculator.scan_completed


def get_calculator() -> PositionCalculator:
    """Calculator 객체 직접 접근"""
    return _position_calculator

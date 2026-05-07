#!/usr/bin/env python3
"""
Run Log Tool (Google Spreadsheet Version)
실험 로그를 Google Spreadsheet에 실시간으로 기록하는 Tool
"""

import os
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from .base_tool import BaseTool
from .config_loader import get_path_config

# ===== 경로 및 설정 (Config from YAML) =====
RUNNUM_FILE = get_path_config("RunNumberFile")
JSON_KEY_FILE = get_path_config("JsonKeyFile")
SPREADSHEET_ID = get_path_config("SpreadsheetId")

class RunLogTool(BaseTool):
    """Google Spreadsheet에 실험 로그를 기록하는 Tool"""
    
    def __init__(self):
        super().__init__(
            name="run_log_tool",
            description="Record experimental logs to Google Spreadsheet"
        )
        self.client = None
        self.sheet = None

    def _authenticate(self):
        """Google Sheets API 인증 및 시트 연결"""
        if self.sheet:
            return True
            
        try:
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
            creds = Credentials.from_service_account_file(str(JSON_KEY_FILE), scopes=scopes)
            self.client = gspread.authorize(creds)
            # URL 대신 ID로 오픈
            spreadsheet = self.client.open_by_key(SPREADSHEET_ID)
            self.sheet = spreadsheet.get_worksheet(0)  # 첫 번째 시트 선택
            return True
        except Exception as e:
            print(f"❌ Google Spreadsheet 인증 실패: {str(e)}")
            return False

    def execute(self, params: Dict[str, Any]) -> str:
        """
        로그 기록 또는 업데이트 실행
        
        Args:
            params:
                - command (str, optional): "write" (새 행 추가) 또는 "update" (기존 행 수정)
                - run_num (str, optional): Run 번호 (update 시 필수, 미입력 시 마지막 행)
                - program (str, optional): 프로그램 이름
                - notes (str, optional): 비고/메모
                - evts, start_time, end_time, config: (write 시 사용)
        """
        if not self._authenticate():
            return "❌ Google Spreadsheet 인증에 실패했습니다."

        command = params.get("command", "write")
        
        if command == "write":
            return self._write_row(params)
        elif command == "update":
            return self._update_row(params)
        elif command == "read":
            return self._read_row(params)
        else:
            return f"❌ 지원하지 않는 명령입니다: {command}"

    def _write_row(self, params: Dict[str, Any]) -> str:
        """새로운 로그 행 추가 (DAQ 호출용)"""
        # Run 번호 가져오기
        run_num = params.get("run_num")
        if not run_num:
            try:
                with open(RUNNUM_FILE, "r") as f:
                    # Run이 종료된 후 runnum.txt가 다음 번호로 업데이트되므로, 
                    # 방금 종료된 Run 정보를 위해 -1을 수행함
                    val = f.read().strip()
                    run_num = str(int(val) - 1)
            except Exception:
                run_num = "Unknown"

        program = params.get("program", "")
        evts = params.get("evts", "")
        start_time = params.get("start_time", "")
        end_time = params.get("end_time", "")
        config = params.get("config", "")
        notes = params.get("notes", "")
        
        # 추가 정보 (Position, Energy 등)
        def safe_round(val, digits=1):
            if val == "" or val is None:
                return ""
            try:
                return round(float(val), digits)
            except (ValueError, TypeError):
                return val

        pos_h = safe_round(params.get("pos_h", ""))
        pos_v = safe_round(params.get("pos_v", ""))
        pos_rot = safe_round(params.get("pos_rot", ""))
        pos_tilt = safe_round(params.get("pos_tilt", ""))
        beam_energy = params.get("beam_energy", "")

        # 실제 시트 컬럼 순서 (스크린샷 기준)
        # B(2): Program | C(3): Run # | D(4): evts | E(5): Time(start) | F(6): Time(end)
        # G(7): HV DRC | H(8): HV Aux | I(9): Pos H | J(10): Pos V | K(11): Pos Rot | L(12): Pos Tilt
        # M(13): Trigger Setup | N(14): Beam Type | O(15): Beam Energy | P(16): Beam Rate
        # Q(17): Config | R(18): Notes
        row = [
            program,        # B: Program
            run_num,        # C: Run #
            evts,           # D: evts
            start_time,     # E: Time (start)
            end_time,       # F: Time (end)
            "",             # G: HV DRC
            "",             # H: HV Aux
            pos_h,          # I: Position H
            pos_v,          # J: Position V
            pos_rot,        # K: Position Rot
            pos_tilt,       # L: Position Tilt
            "",             # M: Trigger Setup
            "",             # N: Beam Type
            beam_energy,    # O: Beam Energy
            "",             # P: Beam Rate
            config,         # Q: Config
            notes           # R: Notes
        ]

        try:
            # 1. 마지막 데이터 행 찾기 (Run #가 있는 C열 기준)
            # col_values(3)은 C열의 모든 값을 가져옵니다.
            col_c_values = self.sheet.col_values(3)
            next_row = len(col_c_values) + 1
            
            # 헤더가 5행까지 있으므로, 데이터는 최소 6행부터 시작해야 함
            if next_row < 6:
                next_row = 6
            
            # 2. 데이터 업데이트 (B열 ~ R열)
            range_label = f"B{next_row}:R{next_row}"
            self.sheet.update(range_label, [row], value_input_option='USER_ENTERED')
            
            return f"✅ 새 Run 로그 추가 완료 (Run: {run_num}, Row: {next_row})"
        except Exception as e:
            return f"❌ 로그 추가 실패: {str(e)}"

    # Header row index (1-based) and column mapping for read
    _HEADER_ROW = 5
    _READ_COLUMNS = {
        2: "Program", 3: "Run #", 4: "Events", 5: "Start", 6: "End",
        7: "HV DRC", 8: "HV Aux", 9: "Pos H", 10: "Pos V",
        11: "Pos Rot", 12: "Pos Tilt", 13: "Trigger", 14: "Beam Type",
        15: "Beam Energy", 16: "Beam Rate", 17: "Config", 18: "Notes",
    }

    def _read_row(self, params: Dict[str, Any]) -> str:
        """Run 번호로 해당 행의 모든 정보를 읽어 반환"""
        run_num = params.get("run_num")
        if not run_num:
            return "⚠️ 조회할 Run 번호를 알려주세요."
        try:
            all_data = self.sheet.get_all_values()
            target_row = None
            for row in all_data:
                if len(row) > 2 and str(row[2]) == str(run_num):
                    target_row = row
                    break
            if target_row is None:
                return f"❌ Run {run_num}을 시트에서 찾을 수 없습니다."

            lines = [f"📋 Run {run_num} 로그:"]
            for col_idx, label in sorted(self._READ_COLUMNS.items()):
                val = target_row[col_idx - 1] if col_idx - 1 < len(target_row) else ""
                if val:
                    lines.append(f"  {label}: {val}")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ 로그 조회 실패: {str(e)}"

    # Column name → (sheet column index, display label)
    UPDATABLE_COLUMNS = {
        "program":       (2,  "Program"),
        "notes":         (18, "Notes"),
        "config":        (17, "Config"),
        "beam_energy":   (15, "Beam Energy"),
        "beam_type":     (14, "Beam Type"),
        "trigger_setup": (13, "Trigger Setup"),
        "hv_drc":        (7,  "HV DRC"),
        "hv_aux":        (8,  "HV Aux"),
    }

    def _update_row(self, params: Dict[str, Any]) -> str:
        """기존 로그 행 업데이트 (UPDATABLE_COLUMNS 에 있는 모든 열 지원)"""
        run_num = params.get("run_num")

        # Collect columns to update: {"column_key": value}
        to_update = {
            col: params[col]
            for col in self.UPDATABLE_COLUMNS
            if params.get(col) is not None
        }

        if not to_update:
            return "⚠️ 업데이트할 정보가 없습니다."

        try:
            all_data = self.sheet.get_all_values()

            if run_num:
                target_row_idx = -1
                for i, row in enumerate(all_data):
                    if len(row) > 2 and str(row[2]) == str(run_num):
                        target_row_idx = i + 1
                        break
            else:
                target_row_idx = len(all_data)
                run_num = all_data[-1][2] if len(all_data[-1]) > 2 else "Last"

            if target_row_idx <= 1:
                return f"❌ Run {run_num}을 시트에서 찾을 수 없습니다."

            updates = []
            for col_key, value in to_update.items():
                col_idx, label = self.UPDATABLE_COLUMNS[col_key]
                self.sheet.update_cell(target_row_idx, col_idx, value)
                updates.append(f"{label}='{value}'")

            return f"✅ Run {run_num} 업데이트 완료: {', '.join(updates)}"

        except Exception as e:
            return f"❌ 로그 업데이트 실패: {str(e)}"

#!/usr/bin/env python3
"""
DAQ Tool

Test-beam DAQ 시스템 제어 Tool
- daq_run: DAQ 실행 (이벤트 수집)
"""

import os
import time
import shutil
import subprocess
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from .base_tool import BaseTool
from .run_log_tool import RunLogTool
from .config_loader import get_path_config, get_data_directory, load_config


# ===== DAQ 설정 (Config from YAML) =====
WORKDIR = get_path_config("DaqWorkDir")
RUN_SCRIPT = get_path_config("DaqScript")
RUNNUM_FILE = get_path_config("RunNumberFile")
KILLME_FILE = get_path_config("KillMeFile")
NOTICE_BASE = get_path_config("NoticeBase")

_studio = load_config()["StudioSSH"]
STUDIO_HOST = _studio["Host"]
STUDIO_USER = _studio["User"]
STUDIO_KEY  = _studio["KeyFile"]


class DAQRunTool(BaseTool):
    """DAQ 실행 Tool"""
    
    def __init__(self):
        super().__init__(
            name="daq_run_tool",
            description="Execute DAQ run with specified events and configuration"
        )
    
    def execute(self, params: Dict[str, Any], line_callback=None) -> str:
        """
        DAQ 실행
        
        Args:
            params:
                - events (int): 수집할 이벤트 수
                - config (str, optional): 설정 파일 (기본값: "setup")
                - pos_h, pos_v, pos_rot, pos_tilt, beam_energy (optional): 로깅용 정보
        
        Returns:
            실행 결과 문자열
        """
        # 파라미터 검증
        valid, error = self.validate_params(params, ["events"])
        if not valid:
            return f"❌ 파라미터 오류: {error}"
        
        events = params["events"]
        config = params.get("config", "setup")
        
        try:
            # events가 정수인지 확인
            events = int(events)
            if events <= 0:
                return "❌ 이벤트 수는 양수여야 합니다"
        except (TypeError, ValueError):
            return f"❌ 잘못된 이벤트 수: {events}"
        
        # Run number 읽기
        try:
            with open(RUNNUM_FILE, "r") as f:
                runnum = f.read().strip()
        except FileNotFoundError:
            return "❌ runnum.txt 파일을 찾을 수 없습니다. DAQ start를 먼저 실행하세요."

        # ── DQM live session: spawn monit --LIVE alongside DAQ ────────────────
        # Best-effort: if anything fails (no manifest, missing binary, etc.)
        # we still run DAQ — DQM is auxiliary.
        dqm_started = False
        try:
            from agents.dqm_live_worker import dqm_session
            from agents.agent_runner import shared_state
            agent_type = shared_state.get("agent_type")
            output_queue = shared_state.get("_output_queue")
            if agent_type and output_queue is not None:
                # Manifest cell templates use ${current_tower} as the bare digit
                # (e.g. fCanvas_Tower5). shared_state stores the prefixed form
                # ("T5"), so strip the leading "T".
                tower_raw = shared_state.get("current_tower") or "5"
                if isinstance(tower_raw, str) and tower_raw.upper().startswith("T"):
                    tower_num = tower_raw[1:]
                else:
                    tower_num = str(tower_raw)
                dqm_session.start(
                    run_number=int(runnum),
                    agent_type=agent_type,
                    output_queue=output_queue,
                    context={
                        "current_tower": tower_num,
                        "current_energy": shared_state.get("current_energy", ""),
                    },
                )
                dqm_started = True
        except Exception as _dqm_err:
            print(f"⚠️ DQM live start skipped: {_dqm_err}", flush=True)

        # Mac Studio에서 SSH로 DAQ 실행
        cmd = (
            f"ssh -i {STUDIO_KEY} -o IdentitiesOnly=yes -o IdentityAgent=none"
            f" -o StrictHostKeyChecking=no -o BatchMode=yes"
            f" {STUDIO_USER}@{STUDIO_HOST}"
            f" 'cd {WORKDIR} && bash {RUN_SCRIPT} {config} {events}'"
        )
        
        try:
            def _emit(line: str):
                """Print and stream to web UI via callback."""
                print(line, flush=True)
                output_lines.append(line)
                if line_callback:
                    line_callback(line)

            output_lines = []
            _emit("🟡 DAQ Run Started")
            _emit(f"📋 Command: {cmd}")
            _emit(f"📝 Run: {runnum} | Config: {config} | Events: {events}")
            _emit("💡 Tip: Create 'KILLME' file to stop execution")
            _emit("")

            start_time = datetime.now()

            # 프로세스 실행 (SSH 원격 실행 - cwd는 SSH 명령 안에서 cd로 처리)
            process = subprocess.Popen(
                cmd,
                shell=True,
                preexec_fn=os.setpgrp,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # stderr를 stdout으로 통합
                text=True,
                bufsize=1  # 라인 버퍼링
            )

            _emit("📄 Execution Output:")
            
            # 실시간으로 출력 읽기 (데드락 방지)
            while True:
                # poll()을 사용하여 프로세스 종료 여부 확인 전 출력 읽기
                if process.stdout:
                    line = process.stdout.readline()
                    if line:
                        clean_line = line.strip()
                        if clean_line:
                            print(clean_line, flush=True)
                            output_lines.append(clean_line)
                            if line_callback:
                                line_callback(clean_line)
                        continue # 더 읽을 게 있을 수 있으므로 계속
                
                if process.poll() is not None:
                    break
                
                # KILLME 파일 모니터링
                if os.path.exists(KILLME_FILE):
                    print("🛑 KILLME file detected - Stopping execution...", flush=True)
                    output_lines.append("🛑 KILLME file detected - Stopping execution...")
                    process.terminate()
                    process.wait()
                    # Remove via SSH — network mount may be read-only from MacBook
                    try:
                        subprocess.run(
                            [
                                "ssh", "-i", STUDIO_KEY,
                                "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none",
                                "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
                                f"{STUDIO_USER}@{STUDIO_HOST}",
                                f"rm -f {WORKDIR}/KILLME",
                            ],
                            timeout=10, check=False,
                        )
                    except Exception:
                        pass
                    output_lines.append("❌ DAQ execution stopped by user")
                    return "\n".join(output_lines)
                
                time.sleep(0.1) # 잠깐 쉬어서 CPU 부하 줄임
            
            # 남은 출력물 마저 읽기
            if process.stdout:
                remaining = process.stdout.read()
                if remaining:
                    for line in remaining.strip().split('\n'):
                        if line.strip():
                            print(line.strip(), flush=True)
                            output_lines.append(line.strip())
            
            process.wait()

            end_time = datetime.now()
            duration = round((end_time - start_time).total_seconds(), 2)
            timestamp = start_time.strftime("%H:%M")

            # 출력 추가 (이미 실시간으로 출력됨)
            output_lines.append("")

            # 디스크 공간 확인
            total, used, free = shutil.disk_usage(".")
            free_gb = round(free / (1024 ** 3), 2)
            percent_used = used / total * 100

            # 실행 결과 확인
            if process.returncode == 0:
                # 실제 기록된 이벤트 수 확인 (FILE_* 전체 카운트)
                # BaseDirectory(config_general.yml)를 단일 진실 소스로 사용 — DQM/HV/dat_plot과 동일.
                try:
                    from pathlib import Path as _Path
                    data_base = get_data_directory()
                    run_wave_dir = (
                        _Path(data_base) / f"Run_{runnum}"
                        / f"Run_{runnum}_Wave"
                        / f"Run_{runnum}_Wave_MID_8"
                    )
                    dat_files = sorted(run_wave_dir.glob(f"Run_{runnum}_Wave_MID_8_FILE_*.dat"))
                    if dat_files:
                        pass
                except Exception:
                    pass

                # 상세 로그 기록 (Google Spreadsheet - RunLogTool 호출)
                log_tool = RunLogTool()
                log_params = {
                    "run_num": runnum,
                    "evts": events,
                    "start_time": start_time.strftime("%H:%M"),
                    "end_time": end_time.strftime("%H:%M"),
                    "config": config,
                    "pos_h": params.get("pos_h", ""),
                    "pos_v": params.get("pos_v", ""),
                    "pos_rot": params.get("pos_rot", ""),
                    "pos_tilt": params.get("pos_tilt", ""),
                    "beam_energy": params.get("beam_energy", ""),
                    "program": params.get("program", ""),
                }
                log_result = log_tool.execute(log_params)
                
                output_lines.append(f"Config: {config} | Events: {events}")
                output_lines.append(f"Duration: {duration}s | Started: {timestamp}")
                output_lines.append(f"Free Space: {free_gb}GB")
                output_lines.append(f"📝 {log_result}")
                
                if percent_used >= 90:
                    output_lines.append("⚠️ Warning: Disk usage exceeds 90%!")
            else:
                # 실패
                output_lines.append(f"❌ DAQ Run Failed (Exit Code: {process.returncode})")
                output_lines.append(f"Attempted Command: {cmd}")
                output_lines.append(f"Duration: {duration}s | Started: {timestamp}")
            
            return "\n".join(output_lines)

        except Exception as e:
            return f"❌ DAQ 실행 중 오류: {str(e)}"
        finally:
            # Always stop DQM live session (touches sentinel, waits for monit
            # to flush a final chunk and exit).
            if dqm_started:
                try:
                    from agents.dqm_live_worker import dqm_session
                    dqm_session.stop()
                except Exception as _dqm_stop_err:
                    print(f"⚠️ DQM live stop error: {_dqm_stop_err}", flush=True)

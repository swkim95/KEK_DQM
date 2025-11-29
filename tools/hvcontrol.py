#!/usr/bin/env python3
"""
HV Control System for CAEN HV Supply (Agent.py 전용)
AI Agent를 통한 자연어 기반 HV 제어 시스템
"""

import json
import shlex
from datetime import datetime
from typing import Dict, Any, List, Tuple

import paramiko
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# 하드코딩된 SSH 설정 (serve_rootfiles.py와 동일 경로/계정 사용)
HV_SSH_CONFIG = {
    'host': '192.168.0.12',
    'port': 22,
    'username': 'yoolab',
    'password': '37326',
    'key_path': None
}

HV_WRAPPER_WORKDIR = "/home/yoolab/Downloads/CAENHVWrapper-6.6/HVWrapperDemo"
HV_CONFIG_FILENAME = "config.txt"
HV_CONFIG_RELATIVE_PATH = f"../config/{HV_CONFIG_FILENAME}"
HV_CONFIG_FULL_PATH = f"/home/yoolab/Downloads/CAENHVWrapper-6.6/config/{HV_CONFIG_FILENAME}"
HV_ENV_PRE_COMMAND = "export LD_LIBRARY_PATH=/usr/lib64/:$LD_LIBRARY_PATH"
DEFAULT_I0SET = "500"


class AIHVCommandParser:
    """AI 기반 자연어 HV 명령어 파서 (Agent.py 전용)"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def parse_natural_language_with_ai(self, text: str) -> Dict[str, Any]:
        """AI를 사용해 자연어를 파싱하여 hvctl 명령어로 변환"""
        
        system_prompt = """
당신은 CAEN HV Supply 제어를 위한 hvctl 명령어 변환 전문가입니다.

사용 가능한 명령어 타입:
- voltage: 채널 전압 설정 (config.txt 수정 후 Pw On)
- on: 채널 전원 켜기 (Pw On)
- off: 채널 전원 끄기 (Pw Off)

중요한 구분:
- 다른 명령어는 허용되지 않습니다.

사용자의 자연어 입력을 분석하여 다음 JSON 형태로만 응답하세요:
{
    "command": "명령어타입만",
    "slot": 슬롯번호(숫자),
    "channels": "채널범위(예: 0,1,2 또는 0-7 또는 전체는 0-23)",
    "voltage": 전압값(숫자),
    "current": 전류값(숫자),
    "ramp_up": 램프업속도(숫자),
    "ramp_down": 램프다운속도(숫자)
}

중요: "command" 필드에는 "voltage", "on", "off" 만 사용하세요. "hvctl"은 포함하지 마세요.
파라미터가 불필요하거나 명시되지 않은 경우 null로 설정하세요.

예시:
- "전체 상태 확인해줘" → {"command": "check", "slot": null, "channels": null, ...}
- "슬롯 2 채널 1 상태 확인해줘" → {"command": "read", "slot": 2, "channels": "1", ...}
- "슬롯 3 채널 전체 켜줘" → {"command": "on", "slot": 3, "channels": "0-23", ...}
"""
        
        try:
            full_prompt = f"{system_prompt}\n\n사용자 입력: {text}\n\n분석 결과를 JSON으로만 응답하세요:"
            response = self.llm.invoke([HumanMessage(content=full_prompt)])
            
            # JSON 추출
            content = response.content.strip()
            
            # JSON 부분만 추출
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].strip()
            
            # JSON 파싱
            result = json.loads(content)
            
            return {
                'success': True,
                'command': result.get('command'),
                'parameters': {k: v for k, v in result.items() if k not in ['command'] and v is not None},
                'original_text': text
            }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'AI 파싱 실패: {str(e)}',
                'original_text': text
            }
    
    
class HVController:
    """HV 컨트롤러 클래스 (Agent.py 전용)"""
    
    def __init__(self, llm):
        self.parser = AIHVCommandParser(llm)
        self.ssh_client = None
    
    def connect_ssh(self) -> bool:
        """SSH 연결 (하드코딩된 설정 사용)"""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # 하드코딩된 SSH 설정 사용
            if HV_SSH_CONFIG['key_path']:
                # SSH 키 사용
                key = paramiko.RSAKey.from_private_key_file(HV_SSH_CONFIG['key_path'])
                self.ssh_client.connect(
                    hostname=HV_SSH_CONFIG['host'],
                    port=HV_SSH_CONFIG['port'],
                    username=HV_SSH_CONFIG['username'],
                    pkey=key
                )
            else:
                # 비밀번호 사용
                self.ssh_client.connect(
                    hostname=HV_SSH_CONFIG['host'],
                    port=HV_SSH_CONFIG['port'],
                    username=HV_SSH_CONFIG['username'],
                    password=HV_SSH_CONFIG['password']
                )
            
            print(f"SSH 연결 성공: {HV_SSH_CONFIG['host']}:{HV_SSH_CONFIG['port']}")
            return True
            
        except Exception as e:
            print(f"SSH 연결 실패: {e}")
            print(f"연결 정보: {HV_SSH_CONFIG['username']}@{HV_SSH_CONFIG['host']}:{HV_SSH_CONFIG['port']}")
            return False
    
    def disconnect_ssh(self):
        """SSH 연결 해제"""
        if self.ssh_client:
            self.ssh_client.close()
            self.ssh_client = None
    
    def ensure_connection(self) -> bool:
        """SSH 연결 상태를 확인하고 필요시 재연결"""
        try:
            if (self.ssh_client and
                self.ssh_client.get_transport() and
                self.ssh_client.get_transport().is_active()):
                return True
        except Exception:
            pass
        
        print("SSH 연결이 없거나 비활성 상태입니다. 재연결 시도...")
        return self.connect_ssh()
    
    def run_remote_command(self, command: str, workdir: str = HV_WRAPPER_WORKDIR) -> Tuple[str, str]:
        """원격 서버에서 명령을 실행"""
        if not self.ensure_connection():
            raise RuntimeError("SSH 연결 실패")
        
        command_segments = []
        if workdir:
            command_segments.append(f"cd {shlex.quote(workdir)}")
        if HV_ENV_PRE_COMMAND:
            command_segments.append(HV_ENV_PRE_COMMAND)
        command_segments.append(command)
        
        remote_cmd = " && ".join(command_segments)
        wrapped = f"bash -lc {shlex.quote(remote_cmd)}"
        
        stdin, stdout, stderr = self.ssh_client.exec_command(wrapped)
        output = stdout.read().decode('utf-8', errors='ignore').strip()
        error = stderr.read().decode('utf-8', errors='ignore').strip()
        return output, error
    
    def read_remote_file(self, remote_path: str) -> str:
        """원격 파일 읽기"""
        if not self.ensure_connection():
            raise RuntimeError("SSH 연결 실패")
        
        sftp = self.ssh_client.open_sftp()
        try:
            with sftp.open(remote_path, 'r') as remote_file:
                data = remote_file.read()
                if isinstance(data, bytes):
                    return data.decode('utf-8', errors='ignore')
                return data
        except IOError:
            return ""
        finally:
            sftp.close()
    
    def write_remote_file(self, remote_path: str, content: str):
        """원격 파일 쓰기 (임시 파일로 안전하게 교체)"""
        if not self.ensure_connection():
            raise RuntimeError("SSH 연결 실패")
        
        temp_path = f"{remote_path}.tmp"
        sftp = self.ssh_client.open_sftp()
        try:
            with sftp.open(temp_path, 'w') as remote_file:
                remote_file.write(content)
            sftp.rename(temp_path, remote_path)
        finally:
            try:
                sftp.remove(temp_path)
            except Exception:
                pass
            sftp.close()
    
    def prepare_plan_from_text(self, natural_language: str) -> Dict[str, Any]:
        """자연어 입력을 받아 config 변경 계획 생성"""
        parsed = self.parser.parse_natural_language_with_ai(natural_language)
        if not parsed.get('success'):
            return {'success': False, 'error': parsed.get('error', '파싱 실패')}
        
        command = (parsed.get('command') or '').lower()
        params = parsed.get('parameters', {})
        
        try:
            if command in ('voltage', 'set_voltage'):
                return self._prepare_set_voltage_plan(params)
            if command in ('on', 'off'):
                return self._prepare_power_plan(command, params)
            return {'success': False, 'error': f"지원되지 않는 명령입니다: {command}"}
        except ValueError as exc:
            return {'success': False, 'error': str(exc)}
        except Exception as exc:
            return {'success': False, 'error': f"계획 생성 실패: {str(exc)}"}
    
    def apply_plan(self, plan: Dict[str, Any]) -> str:
        """확정된 config 계획 적용"""
        action = plan.get('action')
        if action == 'set_voltage':
            return self._apply_set_voltage_plan(plan)
        if action == 'power_toggle':
            return self._apply_power_plan(plan)
        raise ValueError(f"알 수 없는 action 값: {action}")
    
    def _prepare_set_voltage_plan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        slot_value = None
        if params.get('slot') is not None:
            slot_value = self._ensure_slot(params.get('slot'))
        channels = self._ensure_channels(params.get('channels'))
        voltage = self._ensure_float(params.get('voltage'), '전압')
        current = None
        if params.get('current') is not None:
            current = self._ensure_float(params.get('current'), '전류')
        
        rows, warning = self._safe_read_config_rows()
        row_map = {row['ch']: row for row in rows}
        
        summary_lines = ["다음 전압/전류 변경을 준비했습니다:"]
        for ch in channels:
            row = row_map.get(ch)
            name = row['name'] if row else 'N/A'
            prev_v = row['V0Set'] if row else '설정 없음'
            prev_i = row['I0Set'] if row else '설정 없음'
            target_i = self._format_numeric(current) if current is not None else prev_i
            summary_lines.append(
                f"- Ch {ch} ({name}): V {prev_v} → {self._format_numeric(voltage)}, "
                f"I {prev_i} → {target_i}"
            )
        summary_lines.append("적용 시 ./HVWrappdemo --config ../config/config.txt --Pw On 명령이 실행됩니다.")
        if warning:
            summary_lines.append(f"(현재 config.txt를 불러오지 못했습니다: {warning})")
        
        plan = {
            'action': 'set_voltage',
            'channels': channels,
            'voltage': voltage
        }
        if slot_value is not None:
            plan['slot'] = slot_value
        if current is not None:
            plan['current'] = current
        
        return {'success': True, 'plan': plan, 'summary': "\n".join(summary_lines)}
    
    def _prepare_power_plan(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        slot_value = None
        if params.get('slot') is not None:
            slot_value = self._ensure_slot(params.get('slot'))
        channels = self._ensure_channels(params.get('channels'))
        state = 'On' if command == 'on' else 'Off'
        
        rows, warning = self._safe_read_config_rows()
        row_map = {row['ch']: row for row in rows}
        
        summary_lines = [f"다음 채널에 Pw {state} 명령을 실행합니다:"]
        for ch in channels:
            row = row_map.get(ch)
            name = row['name'] if row else 'N/A'
            summary_lines.append(f"- Ch {ch} ({name})")
        summary_lines.append(f"적용 시 ./HVWrappdemo --config ../config/config.txt --Pw {state} 명령이 실행됩니다.")
        if warning:
            summary_lines.append(f"(현재 config.txt를 불러오지 못했습니다: {warning})")
        
        plan = {
            'action': 'power_toggle',
            'channels': channels,
            'state': state
        }
        if slot_value is not None:
            plan['slot'] = slot_value
        
        return {'success': True, 'plan': plan, 'summary': "\n".join(summary_lines)}
    
    def _apply_set_voltage_plan(self, plan: Dict[str, Any]) -> str:
        channels = [int(ch) for ch in plan.get('channels', [])]
        if not channels:
            raise ValueError("채널 정보가 없습니다.")
        voltage = self._ensure_float(plan.get('voltage'), '전압')
        current = None
        if plan.get('current') is not None:
            current = self._ensure_float(plan.get('current'), '전류')
        
        rows = self._read_config_rows()
        row_map = {row['ch']: row for row in rows}
        target_voltage = self._format_numeric(voltage)
        target_current = self._format_numeric(current) if current is not None else None
        
        for ch in channels:
            row = row_map.get(ch)
            if not row:
                row = {'ch': ch, 'name': f"CH{ch}", 'V0Set': '0', 'I0Set': DEFAULT_I0SET}
                rows.append(row)
                row_map[ch] = row
            row['V0Set'] = target_voltage
            if target_current is not None:
                row['I0Set'] = target_current
        
        self._write_config_rows(rows)
        
        stdout, stderr = self.run_remote_command(
            f"./HVWrappdemo --config {HV_CONFIG_RELATIVE_PATH} --Pw On"
        )
        if stderr:
            raise RuntimeError(f"원격 명령 실패: {stderr}")
        
        channel_list = ", ".join(map(str, channels))
        return (
            "✅ 전압 설정 적용 완료!\n"
            f"- 채널: {channel_list}\n"
            f"- 목표 전압: {target_voltage} V"
            f"{f', 전류: {target_current} μA' if target_current is not None else ''}\n"
            f"- 실행 명령: ./HVWrappdemo --config ../config/config.txt --Pw On\n"
            f"- 결과: {stdout or '(출력 없음)'}\n"
            f"- 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    
    def _apply_power_plan(self, plan: Dict[str, Any]) -> str:
        channels = [int(ch) for ch in plan.get('channels', [])]
        if not channels:
            raise ValueError("채널 정보가 없습니다.")
        state = str(plan.get('state', 'On')).strip().capitalize()
        if state not in {'On', 'Off'}:
            raise ValueError("Pw 상태는 On 또는 Off만 지원합니다.")
        
        stdout, stderr = self.run_remote_command(
            f"./HVWrappdemo --config {HV_CONFIG_RELATIVE_PATH} --Pw {state}"
        )
        if stderr:
            raise RuntimeError(f"원격 명령 실패: {stderr}")
        
        channel_list = ", ".join(map(str, channels))
        return (
            f"✅ Pw {state} 적용 완료!\n"
            f"- 채널: {channel_list}\n"
            f"- 실행 명령: ./HVWrappdemo --config ../config/config.txt --Pw {state}\n"
            f"- 결과: {stdout or '(출력 없음)'}\n"
            f"- 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    
    def _ensure_slot(self, value: Any) -> int:
        if value is None:
            raise ValueError("슬롯 번호가 필요합니다.")
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ValueError(f"잘못된 슬롯 번호: {value}")
    
    def _ensure_channels(self, raw_channels: Any) -> List[int]:
        if raw_channels is None:
            raise ValueError("채널 정보가 필요합니다.")
        expr = str(raw_channels).strip()
        if expr.lower() in {'전체', 'all'}:
            expr = '0-23'
        channels = self._expand_channels(expr)
        if not channels:
            raise ValueError("유효한 채널을 찾지 못했습니다.")
        return channels
    
    def _ensure_float(self, value: Any, label: str) -> float:
        if value is None:
            raise ValueError(f"{label} 값이 필요합니다.")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} 값이 잘못되었습니다: {value}")
    
    def _expand_channels(self, channels_expr: str) -> List[int]:
        """채널 표현(0,1,2 또는 0-7)을 리스트로 확장"""
        if not channels_expr:
            return []
        
        channels: List[int] = []
        for part in channels_expr.split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                start_str, end_str = part.split('-', 1)
                start = int(start_str)
                end = int(end_str)
                if start > end:
                    start, end = end, start
                channels.extend(range(start, end + 1))
            else:
                channels.append(int(part))
        
        # 중복 제거 및 정렬
        return sorted(set(channels))
    
    def _safe_read_config_rows(self) -> Tuple[List[Dict[str, Any]], str]:
        try:
            return self._read_config_rows(), ""
        except Exception as exc:
            return [], str(exc)
    
    def _read_config_rows(self) -> List[Dict[str, Any]]:
        content = self.read_remote_file(HV_CONFIG_FULL_PATH)
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return []
        
        rows: List[Dict[str, Any]] = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                ch = int(parts[0])
            except ValueError:
                continue
            name = parts[1]
            vset = parts[2]
            iset = parts[3]
            rows.append({'ch': ch, 'name': name, 'V0Set': vset, 'I0Set': iset})
        return rows
    
    def _write_config_rows(self, rows: List[Dict[str, Any]]):
        lines = ["ch name V0Set I0Set", ""]
        for row in sorted(rows, key=lambda r: r['ch']):
            name = row.get('name', 'None')
            v0 = row.get('V0Set', '0')
            i0 = row.get('I0Set', DEFAULT_I0SET)
            lines.append(f"{row['ch']} {name} {v0} {i0}")
        self.write_remote_file(HV_CONFIG_FULL_PATH, "\n".join(lines) + "\n")
    
    def _format_numeric(self, value: Any) -> str:
        try:
            num = float(value)
            if abs(num - int(num)) < 1e-6:
                return str(int(num))
            return f"{num}"
        except (TypeError, ValueError):
            return str(value)
    
    def execute_command_with_ai(self, natural_language: str) -> str:
        """AI를 사용해 자연어 명령어를 파싱하고 plan 정보를 CONFIRM_COMMAND로 반환"""
        try:
            plan_result = self.prepare_plan_from_text(natural_language)
            if not plan_result.get('success'):
                return f"❌ HV 명령 파싱 실패: {plan_result.get('error', '알 수 없는 오류')}"
            
            summary = plan_result.get('summary', '변경 내용을 확인해주세요.')
            summary += "\n\n⚠️ 반드시 인간 운영자 확인 후 실행하세요."
            payload = json.dumps(plan_result['plan'], ensure_ascii=False)
            return f"{summary}\n\nCONFIRM_COMMAND:hv_config:{payload}"
        except Exception as e:
            return f"❌ HV 명령 처리 오류: {str(e)}"
    


# Agent.py 통합용 Tool 함수들
# 전역 컨트롤러 인스턴스 
_hv_controller = None

def get_hv_controller(llm):
    """HV 컨트롤러 싱글톤 인스턴스 반환"""
    global _hv_controller
    if _hv_controller is None:
        _hv_controller = HVController(llm)
    return _hv_controller

@tool
def hv_execute_tool(command: str) -> str:
    """
    AI를 사용해 자연어 HV 명령을 해석하고 확인을 요청합니다.
    
    Args:
        command: 자연어로 된 HV 제어 명령 
                예: "슬롯 3 채널 0,1,2 전압을 1500V로 설정해줘"
                    "상태 확인해줘"
                    "슬롯 2 채널 전체 켜줘"
    
    Returns:
        CONFIRM_COMMAND 또는 오류 메시지
    """
    try:
        from llm_provider import get_llm
        llm = get_llm()
        controller = get_hv_controller(llm)
        return controller.execute_command_with_ai(command)
    except Exception as e:
        return f"❌ HV 제어 시스템 오류: {str(e)}"

@tool
def hv_confirm_command_tool(command_id: str) -> str:
    """
    확인된 HV 명령을 실행합니다.
    
    Args:
        command_id: 명령 식별자 (예: "hv_config:{\"action\":\"set_voltage\",...}")
    
    Returns:
        명령 실행 결과
    """
    try:
        if ':' in command_id:
            command_type, payload = command_id.split(':', 1)
        else:
            command_type, payload = "hv_config", command_id
        
        from llm_provider import get_llm
        llm = get_llm()
        controller = get_hv_controller(llm)
        
        if command_type != "hv_config":
            return f"❌ 잘못된 명령 타입: {command_type}"
        
        try:
            plan = json.loads(payload)
        except json.JSONDecodeError:
            return "❌ 명령 데이터를 해석할 수 없습니다."
        return controller.apply_plan(plan)
        
    except Exception as e:
        return f"❌ HV 명령 실행 오류: {str(e)}"

#!/usr/bin/env python3

import os
import sys
import signal
import subprocess
import json
import fcntl
import select
import threading
import shlex
from datetime import datetime

import paramiko
from flask import Flask, request, Response, jsonify, send_from_directory, send_file
from flask_cors import CORS

# 현재 실행되는 파이썬 파일의 디렉토리를 기준으로 절대경로 생성
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "output")
INDEX_DIR = BASE_DIR
PROJECT_ROOT = os.path.dirname(BASE_DIR)

TOOLS_DIR = os.path.join(PROJECT_ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.append(TOOLS_DIR)

DEFAULT_HV_SSH_CONFIG = {
    'host': '192.168.0.12',
    'port': 22,
    'username': 'yoolab',
    'password': '37326',
    'key_path': None,
}

try:
    from hvcontrol import HV_SSH_CONFIG as IMPORTED_HV_SSH_CONFIG  # type: ignore
    HV_MONITOR_SSH_CONFIG = IMPORTED_HV_SSH_CONFIG or DEFAULT_HV_SSH_CONFIG
except Exception:
    HV_MONITOR_SSH_CONFIG = DEFAULT_HV_SSH_CONFIG

app = Flask(__name__)
CORS(app)  # CORS 활성화 (Agent 창에서 API 호출 허용)

# 환경변수 관리를 위한 전역 변수
ENV_INITIALIZED = False
CUSTOM_ENV = os.environ.copy()

# 실행 중인 프로세스 추적
current_process = None
process_lock = threading.Lock()

# 백그라운드 anomaly detection 프로세스 추적
anomaly_detection_processes = {}  # {run_number: subprocess}
anomaly_lock = threading.Lock()

# AI Agent 프로세스 추적
agent_process = None
agent_lock = threading.Lock()

# HV Monitor 상태
hv_monitor_client = None
hv_monitor_lock = threading.Lock()
HV_MONITOR_WORKING_DIR = "/home/yoolab/Downloads/CAENHVWrapper-6.6/HVWrapperDemo"
HV_MONITOR_STATUS_COMMAND = "./HVWrappdemo --ch all --get Status"
HV_MONITOR_MODE_COMMANDS = {
    'vmon': "./HVWrappdemo --ch all --VMon",
    'imon': "./HVWrappdemo --ch all --IMon",
}
HV_MONITOR_DEFAULT_COMMAND = HV_MONITOR_MODE_COMMANDS['imon']
HV_MONITOR_PRE_COMMAND = "export LD_LIBRARY_PATH=/usr/lib64/:$LD_LIBRARY_PATH"


def ensure_hv_monitor_client():
    """HV 모니터용 SSH 클라이언트를 확보합니다."""
    global hv_monitor_client
    with hv_monitor_lock:
        # 기존 연결이 살아있는지 확인
        if hv_monitor_client:
            try:
                transport = hv_monitor_client.get_transport()
                if transport and transport.is_active():
                    return hv_monitor_client
            except Exception:
                pass
            # 비정상인 경우 정리
            try:
                hv_monitor_client.close()
            except Exception:
                pass
            hv_monitor_client = None

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            if HV_MONITOR_SSH_CONFIG.get('key_path'):
                key = paramiko.RSAKey.from_private_key_file(HV_MONITOR_SSH_CONFIG['key_path'])
                client.connect(
                    hostname=HV_MONITOR_SSH_CONFIG['host'],
                    port=HV_MONITOR_SSH_CONFIG.get('port', 22),
                    username=HV_MONITOR_SSH_CONFIG['username'],
                    pkey=key,
                    timeout=10
                )
            else:
                client.connect(
                    hostname=HV_MONITOR_SSH_CONFIG['host'],
                    port=HV_MONITOR_SSH_CONFIG.get('port', 22),
                    username=HV_MONITOR_SSH_CONFIG['username'],
                    password=HV_MONITOR_SSH_CONFIG.get('password'),
                    timeout=10
                )
        except Exception as exc:
            raise RuntimeError(f"HV 모니터 SSH 연결 실패: {exc}") from exc

        hv_monitor_client = client

        return hv_monitor_client


def close_hv_monitor_client():
    """HV 모니터용 SSH 연결을 닫습니다."""
    global hv_monitor_client
    with hv_monitor_lock:
        if hv_monitor_client:
            try:
                hv_monitor_client.close()
            except Exception:
                pass
            hv_monitor_client = None


def run_hv_monitor_command(command: str):
    """윈도우 노트북으로 HVWrappdemo 명령을 전달하고 결과를 반환합니다."""
    cmd = (command or HV_MONITOR_DEFAULT_COMMAND).strip()
    if not cmd:
        cmd = HV_MONITOR_DEFAULT_COMMAND

    last_error = None
    for _ in range(2):
        try:
            client = ensure_hv_monitor_client()
            command_segments = []
            if HV_MONITOR_WORKING_DIR:
                command_segments.append(f"cd {shlex.quote(HV_MONITOR_WORKING_DIR)}")
            if HV_MONITOR_PRE_COMMAND:
                command_segments.append(HV_MONITOR_PRE_COMMAND)
            command_segments.append(cmd)
            combined = " && ".join(command_segments)
            remote_cmd = f"bash -lc {shlex.quote(combined)}"
            stdin, stdout, stderr = client.exec_command(remote_cmd)
            output = stdout.read().decode('utf-8', errors='ignore').strip()
            error = stderr.read().decode('utf-8', errors='ignore').strip()
            success = len(error) == 0
            return {
                'command': cmd,
                'stdout': output,
                'stderr': error,
                'success': success
            }
        except Exception as exc:
            last_error = str(exc)
            close_hv_monitor_client()

    raise RuntimeError(last_error or "HV 모니터 명령 실행 실패")

def parse_envset_sh():
    """envset.sh 파일을 파싱하여 환경변수를 추출합니다."""
    global ENV_INITIALIZED, CUSTOM_ENV
    
    try:
        envset_path = os.path.join(os.path.dirname(BASE_DIR), "envset.sh")
        
        if not os.path.exists(envset_path):
            return False, "envset.sh file not found"
        
        # 더 포괄적인 환경설정 스크립트 실행
        setup_script = f"""
        cd {os.path.dirname(BASE_DIR)}
        source envset.sh
        
        # ROOT 환경변수들이 제대로 설정되었는지 확인하고 출력
        echo "=== ENVIRONMENT VARIABLES ==="
        env | grep -E "(DYLD|LD_LIBRARY|PATH|ROOT|INSTALL_DIR)" | sort
        echo "=== END ENVIRONMENT ==="
        """
        
        result = subprocess.run(
            setup_script,
            shell=True,
            capture_output=True,
            text=True,
            executable='/bin/bash'
        )
        
        if result.returncode != 0:
            return False, f"Failed to source envset.sh: {result.stderr}"
        
        # 환경변수 파싱 - 더 정확한 파싱
        env_section = False
        for line in result.stdout.split('\n'):
            if line.strip() == "=== ENVIRONMENT VARIABLES ===":
                env_section = True
                continue
            elif line.strip() == "=== END ENVIRONMENT ===":
                env_section = False
                continue
            elif env_section and '=' in line:
                key, value = line.split('=', 1)
                CUSTOM_ENV[key] = value
        
        # ROOT 환경변수들을 명시적으로 설정
        if 'ROOTSYS' not in CUSTOM_ENV:
            # Homebrew ROOT의 정확한 ROOTSYS 경로 설정
            try:
                root_cellar_path = "/opt/homebrew/Cellar/root"
                if os.path.exists(root_cellar_path):
                    versions = [d for d in os.listdir(root_cellar_path) if os.path.isdir(os.path.join(root_cellar_path, d))]
                    if versions:
                        latest_version = sorted(versions)[-1]
                        CUSTOM_ENV['ROOTSYS'] = f"/opt/homebrew/Cellar/root/{latest_version}"
                    else:
                        CUSTOM_ENV['ROOTSYS'] = '/opt/homebrew'
                else:
                    CUSTOM_ENV['ROOTSYS'] = '/opt/homebrew'
            except Exception:
                CUSTOM_ENV['ROOTSYS'] = '/opt/homebrew'
        
        # 라이브러리 경로들 설정
        install_lib_path = os.path.join(os.path.dirname(BASE_DIR), "install", "lib")
        
        # ROOT 라이브러리 경로를 동적으로 찾기
        root_lib_path = "/opt/homebrew/lib/root"  # 기본값
        try:
            # ROOT 버전 찾기
            root_cellar_path = "/opt/homebrew/Cellar/root"
            if os.path.exists(root_cellar_path):
                versions = [d for d in os.listdir(root_cellar_path) if os.path.isdir(os.path.join(root_cellar_path, d))]
                if versions:
                    latest_version = sorted(versions)[-1]  # 가장 최신 버전 사용
                    potential_root_lib = f"/opt/homebrew/Cellar/root/{latest_version}/lib/root"
                    if os.path.exists(potential_root_lib):
                        root_lib_path = potential_root_lib
        except Exception:
            pass  # 기본값 사용
        
        # DYLD_LIBRARY_PATH 설정 (install/lib을 최우선으로)
        dyld_paths = [install_lib_path]  # install/lib을 먼저 추가
        if os.path.exists(root_lib_path):
            dyld_paths.append(root_lib_path)
        # 추가 표준 경로들
        additional_paths = ["/opt/homebrew/lib", "/usr/local/lib", "/usr/lib"]
        for path in additional_paths:
            if os.path.exists(path) and path not in dyld_paths:
                dyld_paths.append(path)
        
        if 'DYLD_LIBRARY_PATH' in CUSTOM_ENV:
            existing_paths = CUSTOM_ENV['DYLD_LIBRARY_PATH'].split(':')
            # install_lib_path를 맨 앞에 놓기
            new_paths = [install_lib_path]
            for path in existing_paths + dyld_paths[1:]:
                if path and path not in new_paths:
                    new_paths.append(path)
            CUSTOM_ENV['DYLD_LIBRARY_PATH'] = ':'.join(new_paths)
        else:
            CUSTOM_ENV['DYLD_LIBRARY_PATH'] = ':'.join(dyld_paths)
            
        # DYLD_FALLBACK_LIBRARY_PATH 설정 (macOS에서 더 안정적)
        fallback_paths = [install_lib_path, root_lib_path, "/opt/homebrew/lib", "/usr/lib"]
        if 'DYLD_FALLBACK_LIBRARY_PATH' in CUSTOM_ENV:
            existing_paths = CUSTOM_ENV['DYLD_FALLBACK_LIBRARY_PATH'].split(':')
            for path in fallback_paths:
                if path not in existing_paths:
                    existing_paths.append(path)
            CUSTOM_ENV['DYLD_FALLBACK_LIBRARY_PATH'] = ':'.join(existing_paths)
        else:
            CUSTOM_ENV['DYLD_FALLBACK_LIBRARY_PATH'] = ':'.join(fallback_paths)
            
        # LD_LIBRARY_PATH 설정
        ld_paths = [root_lib_path, install_lib_path, "/opt/homebrew/lib"]
        if 'LD_LIBRARY_PATH' in CUSTOM_ENV:
            existing_paths = CUSTOM_ENV['LD_LIBRARY_PATH'].split(':')
            for path in ld_paths:
                if path not in existing_paths:
                    existing_paths.append(path)
            CUSTOM_ENV['LD_LIBRARY_PATH'] = ':'.join(existing_paths)
        else:
            CUSTOM_ENV['LD_LIBRARY_PATH'] = ':'.join(ld_paths)
        
        # PATH에 ROOT bin 디렉토리 추가
        root_bin_paths = ["/opt/homebrew/bin"]
        
        # ROOTSYS가 설정되어 있다면 해당 bin 디렉토리도 추가
        if 'ROOTSYS' in CUSTOM_ENV:
            potential_root_bin = os.path.join(CUSTOM_ENV['ROOTSYS'], 'bin')
            if os.path.exists(potential_root_bin):
                root_bin_paths.insert(0, potential_root_bin)
        
        if 'PATH' in CUSTOM_ENV:
            current_paths = CUSTOM_ENV['PATH'].split(':')
            for root_bin_path in root_bin_paths:
                if root_bin_path not in current_paths:
                    current_paths.insert(0, root_bin_path)
            CUSTOM_ENV['PATH'] = ':'.join(current_paths)
        else:
            CUSTOM_ENV['PATH'] = f"{':'.join(root_bin_paths)}:{os.environ.get('PATH', '')}"
        
        ENV_INITIALIZED = True
        return True, "Environment initialized successfully"
        
    except Exception as e:
        return False, f"Error parsing envset.sh: {str(e)}"

def check_library_dependencies():
    """라이브러리 의존성을 체크합니다."""
    try:
        install_lib_path = os.path.join(os.path.dirname(BASE_DIR), "install", "lib")
        libdrc_path = os.path.join(install_lib_path, "libdrcTB.dylib")
        monit_path = os.path.join(BASE_DIR, "monit")
        
        checks = []
        
        # 1. 라이브러리 파일 존재 체크
        if os.path.exists(libdrc_path):
            checks.append("✅ libdrcTB.dylib found")
        else:
            checks.append("❌ libdrcTB.dylib NOT found")
        
        # 2. monit 실행파일 존재 체크
        if os.path.exists(monit_path):
            checks.append("✅ monit executable found")
            
            # 3. otool로 의존성 체크 (macOS specific)
            try:
                result = subprocess.run(
                    ['otool', '-L', monit_path],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'libdrcTB.dylib' in line:
                            checks.append(f"📋 monit dependency: {line.strip()}")
                            break
                    else:
                        checks.append("⚠️  libdrcTB.dylib dependency not found in monit")
                else:
                    checks.append("⚠️  Could not check monit dependencies")
            except FileNotFoundError:
                checks.append("⚠️  otool not available (dependency check skipped)")
                
        else:
            checks.append("❌ monit executable NOT found")
        
        # 4. ROOT 라이브러리 경로 체크
        # ROOT 라이브러리 경로를 동적으로 찾기
        root_lib_path = "/opt/homebrew/lib/root"  # 기본값
        try:
            root_cellar_path = "/opt/homebrew/Cellar/root"
            if os.path.exists(root_cellar_path):
                versions = [d for d in os.listdir(root_cellar_path) if os.path.isdir(os.path.join(root_cellar_path, d))]
                if versions:
                    latest_version = sorted(versions)[-1]
                    potential_root_lib = f"/opt/homebrew/Cellar/root/{latest_version}/lib/root"
                    if os.path.exists(potential_root_lib):
                        root_lib_path = potential_root_lib
        except Exception:
            pass
            
        if os.path.exists(root_lib_path):
            checks.append(f"✅ ROOT library directory found: {root_lib_path}")
        else:
            checks.append(f"❌ ROOT library directory NOT found: {root_lib_path}")
        
        # 5. 환경변수 경로 체크
        if 'DYLD_LIBRARY_PATH' in CUSTOM_ENV:
            if install_lib_path in CUSTOM_ENV['DYLD_LIBRARY_PATH'] and root_lib_path in CUSTOM_ENV['DYLD_LIBRARY_PATH']:
                checks.append("✅ DYLD_LIBRARY_PATH includes both install/lib and ROOT/lib")
            elif install_lib_path in CUSTOM_ENV['DYLD_LIBRARY_PATH']:
                checks.append("⚠️  DYLD_LIBRARY_PATH includes install/lib but missing ROOT/lib")
            elif root_lib_path in CUSTOM_ENV['DYLD_LIBRARY_PATH']:
                checks.append("⚠️  DYLD_LIBRARY_PATH includes ROOT/lib but missing install/lib")
            else:
                checks.append("❌ DYLD_LIBRARY_PATH missing both install/lib and ROOT/lib")
        else:
            checks.append("❌ DYLD_LIBRARY_PATH not set")
            
        if 'DYLD_FALLBACK_LIBRARY_PATH' in CUSTOM_ENV:
            if install_lib_path in CUSTOM_ENV['DYLD_FALLBACK_LIBRARY_PATH'] and root_lib_path in CUSTOM_ENV['DYLD_FALLBACK_LIBRARY_PATH']:
                checks.append("✅ DYLD_FALLBACK_LIBRARY_PATH includes both install/lib and ROOT/lib")
            else:
                checks.append("⚠️  DYLD_FALLBACK_LIBRARY_PATH partially configured")
        else:
            checks.append("❌ DYLD_FALLBACK_LIBRARY_PATH not set")
            
        # 6. ROOTSYS 체크
        if 'ROOTSYS' in CUSTOM_ENV:
            checks.append(f"✅ ROOTSYS set to: {CUSTOM_ENV['ROOTSYS']}")
        else:
            checks.append("❌ ROOTSYS not set")
        
        return '\n'.join(checks)
        
    except Exception as e:
        return f"Error checking dependencies: {str(e)}"

def run_command_with_env(command, cwd=None):
    """환경변수를 포함하여 명령어를 실행합니다."""
    if cwd is None:
        cwd = BASE_DIR
        
    # 환경변수가 초기화되지 않았다면 초기화 시도
    if not ENV_INITIALIZED:
        success, message = parse_envset_sh()
        if not success:
            return subprocess.CompletedProcess(
                args=command,
                returncode=1,
                stdout="",
                stderr=f"Environment not initialized: {message}"
            )
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=CUSTOM_ENV,
            executable='/bin/bash'
        )
        return result
    except Exception as e:
        return subprocess.CompletedProcess(
            args=command,
            returncode=1,
            stdout="",
            stderr=str(e)
        )

@app.route('/')
def serve_index():
    return send_from_directory(INDEX_DIR, 'index.html')

def send_file_with_range_support(file_path):
    """Range 요청을 지원하는 파일 전송 함수"""
    def generate():
        with open(file_path, 'rb') as f:
            data = f.read(1024)
            while data:
                yield data
                data = f.read(1024)
    
    file_size = os.path.getsize(file_path)
    range_header = request.headers.get('Range', None)
    
    if not range_header:
        return Response(generate(), 
                       mimetype="application/octet-stream",
                       headers={'Content-Length': str(file_size),
                               'Accept-Ranges': 'bytes'})
    
    # Range 요청 처리 - 다중 범위 처리 개선
    try:
        byte_start = 0
        byte_end = file_size - 1
        
        if range_header:
            range_value = range_header.replace('bytes=', '')
            # 다중 범위인 경우 첫 번째만 사용
            if ',' in range_value:
                range_value = range_value.split(',')[0]
            
            if '-' in range_value:
                parts = range_value.split('-')
                if parts[0]:
                    byte_start = int(parts[0])
                if parts[1]:
                    byte_end = int(parts[1])
        
        # 범위 검증
        if byte_start >= file_size or byte_end >= file_size or byte_start > byte_end:
            return Response("416 Range Not Satisfiable", status=416)
            
        content_length = byte_end - byte_start + 1
        
        def generate_range():
            with open(file_path, 'rb') as f:
                f.seek(byte_start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(1024, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    yield data
                    remaining -= len(data)
        
        return Response(generate_range(),
                       206,  # Partial Content
                       mimetype="application/octet-stream",
                       headers={
                           'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                           'Accept-Ranges': 'bytes',
                           'Content-Length': str(content_length)
                       })
    except Exception as e:
        # Fallback to full file
        return Response(generate(), 
                       mimetype="application/octet-stream",
                       headers={'Content-Length': str(file_size),
                               'Accept-Ranges': 'bytes'})

@app.route('/output/<path:filename>')
def serve_output_file(filename):
    try:
        file_path = os.path.join('output', filename)
        if not os.path.exists(file_path):
            return f"File {filename} not found", 404
            
        # Add cache control headers for ROOT files to ensure fresh content
        response = send_from_directory('output', filename)
        if filename.endswith('.root'):
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            
        return response
    except Exception as e:
        return f"Error serving file {filename}: {str(e)}", 500

@app.route('/files')
def list_root_files():
    try:
        files_info = []
        for f in os.listdir(ROOT_DIR):
            # anomaly_live 관련 JSON/LOG 파일은 제외 (내부용)
            if 'anomaly_live.json' in f or 'anomaly_live.log' in f:
                continue
            
            # ROOT 파일, GIF 파일, JSON 파일, PNG 파일 모두 포함
            if f.endswith((".root", ".gif", ".json", ".png")):
                file_path = os.path.join(ROOT_DIR, f)
                if os.path.exists(file_path):
                    # Get file creation/modification time
                    stat = os.stat(file_path)
                    # Use modification time (mtime) as it's more reliable than creation time
                    modification_time = stat.st_mtime
                    
                    # Determine file type
                    if f.endswith(".gif"):
                        file_type = "gif"
                    elif f.endswith(".json"):
                        file_type = "json"
                    elif f.endswith(".png"):
                        file_type = "png"
                    else:
                        file_type = "root"
                    
                    files_info.append({
                        'name': f,
                        'mtime': modification_time,
                        'size': stat.st_size,
                        'type': file_type
                    })
        
        # Sort by modification time (most recent first)
        files_info.sort(key=lambda x: x['mtime'], reverse=True)
        
        return jsonify(files_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# HV 모니터 페이지
@app.route('/hv-monitor')
def hv_monitor_page():
    return send_from_directory(INDEX_DIR, 'hv_monitor.html')


# 정적 파일들을 서빙하는 라우트 추가
@app.route('/<path:filename>')
def serve_static_files(filename):
    return send_from_directory(INDEX_DIR, filename)


@app.route('/hv-monitor/status', methods=['GET'])
def hv_monitor_status():
    """HVWrappdemo 명령을 실행하여 실시간 측정값과 상태를 반환합니다."""
    requested_mode = request.args.get('mode', 'vmon').strip().lower()
    measurement_command = request.args.get('command')
    hv_log_dir = os.path.join(os.path.dirname(BASE_DIR), 'logs')
    os.makedirs(hv_log_dir, exist_ok=True)
    hv_log_path = os.path.join(hv_log_dir, 'hv_monitor.log')

    def log_execution(entry):
        with open(hv_log_path, 'a') as log_file:
            log_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

    if not measurement_command:
        measurement_command = HV_MONITOR_MODE_COMMANDS.get(requested_mode, HV_MONITOR_DEFAULT_COMMAND)
    else:
        measurement_command = measurement_command.strip() or HV_MONITOR_DEFAULT_COMMAND

    timestamp = datetime.utcnow().isoformat() + 'Z'

    try:
        status_result = run_hv_monitor_command(HV_MONITOR_STATUS_COMMAND)
        measurement_result = run_hv_monitor_command(measurement_command)

        log_entry = {
            'timestamp': timestamp,
            'mode': requested_mode,
            'status_command': HV_MONITOR_STATUS_COMMAND,
            'measurement_command': measurement_command,
            'status': status_result,
            'measurement': measurement_result
        }
        log_execution(log_entry)

        return jsonify({
            'success': status_result['success'] and measurement_result['success'],
            'mode': requested_mode if requested_mode in HV_MONITOR_MODE_COMMANDS else 'custom',
            'timestamp': timestamp,
            'status': status_result,
            'measurement': measurement_result
        })
    except Exception as exc:
        error_entry = {
            'timestamp': timestamp,
            'mode': requested_mode,
            'measurement_command': measurement_command,
            'error': str(exc)
        }
        log_execution(error_entry)
        return jsonify({
            'success': False,
            'error': str(exc)
        }), 500


@app.route('/hv-monitor/disconnect', methods=['POST'])
def hv_monitor_disconnect():
    """HV 모니터링 연결을 종료합니다."""
    close_hv_monitor_client()
    return jsonify({'success': True})

# 명령어 실행 API 추가
@app.route('/execute', methods=['POST'])
def execute_command():
    try:
        data = request.get_json()
        command = data.get('command', '')
        
        if not command:
            return jsonify({'error': 'No command provided'}), 400
        
        # 보안을 위해 특정 명령어만 금지 (rm, cd는 위험할 수 있음)
        command_parts = command.split()
        forbidden_commands = ['rm', 'cd']
        
        if command_parts and command_parts[0] in forbidden_commands:
            return jsonify({'error': f'Command not allowed for security reasons. Forbidden: {", ".join(forbidden_commands)}'}), 403
        
        # 명령어 실행
        result = run_command_with_env(command)
        
        return jsonify({
            'command': command,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Anomaly Detection 실행 API
def execute_anomaly_detection(command):
    """
    Anomaly detection 실행 - detect_run.py 호출
    명령어에서 --RunNumber, --module (tower), --type 등을 파싱
    """
    try:
        # 명령어 파싱
        parts = command.split()
        run_number = None
        towers_input = None
        type_val = None
        
        for i, part in enumerate(parts):
            if part == '--RunNumber' and i + 1 < len(parts):
                run_number = parts[i + 1]
            elif part == '--towers' and i + 1 < len(parts):
                towers_input = parts[i + 1]
            elif part == '--type' and i + 1 < len(parts):
                type_val = parts[i + 1]
        
        if not run_number:
            return "data: " + json.dumps({'type': 'error', 'content': 'Run number not specified'}) + "\n\n", 400
        
        # Tower 리스트 결정
        # --towers가 지정되면 해당 tower들 사용, 아니면 모든 tower (T1 ~ T9)
        if towers_input:
            # towers_input: "T1,T2,T3" 또는 "T1-C" 형식
            # 쉼표로 분리된 경우 각각 tower 추출
            tower_list = []
            for tower_item in towers_input.split(','):
                tower_item = tower_item.strip()
                # "-C" 또는 "-S" 제거하여 tower 이름만 추출
                tower_name = tower_item.split('-')[0] if '-' in tower_item else tower_item
                if tower_name and tower_name not in tower_list:
                    tower_list.append(tower_name)
            towers = tower_list if tower_list else ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']
        else:
            # 모든 tower 사용
            towers = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']
        
        tower_str = ','.join(towers)
        
        def generate():
            yield f"data: {json.dumps({'type': 'output', 'content': '🔍 Starting Anomaly Detection...'})}\n\n"
            yield f"data: {json.dumps({'type': 'output', 'content': f'Run Number: {run_number}'})}\n\n"
            yield f"data: {json.dumps({'type': 'output', 'content': f'Towers: {tower_str}'})}\n\n"
            yield f"data: {json.dumps({'type': 'output', 'content': ''})}\n\n"
            
            # detect_run.py 경로
            detect_script = os.path.join(os.path.dirname(BASE_DIR), 'anomaly', 'detect_run.py')
            
            if not os.path.exists(detect_script):
                yield f"data: {json.dumps({'type': 'error', 'content': f'Anomaly detection script not found: {detect_script}'})}\n\n"
                return
            
            # Python 명령어 구성
            python_cmd = f"cd {os.path.dirname(BASE_DIR)}/anomaly && python3 detect_run.py --run {run_number} --towers {tower_str} --output {os.path.join(BASE_DIR, 'output')}"
            
            yield f"data: {json.dumps({'type': 'command', 'content': f'$ {python_cmd}'})}\n\n"
            
            global current_process
            
            try:
                with process_lock:
                    current_process = subprocess.Popen(
                        python_cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                        cwd=os.path.dirname(BASE_DIR),
                        env=CUSTOM_ENV
                    )
                
                # 실시간 출력 스트리밍
                buffer = ""
                
                while True:
                    if current_process.poll() is not None:
                        break
                    
                    try:
                        char = current_process.stdout.read(1)
                        if not char:
                            continue
                        
                        buffer += char
                        
                        if char == '\n':
                            if buffer and buffer.strip():
                                content = buffer.rstrip('\n')
                                yield f"data: {json.dumps({'type': 'output', 'content': content})}\n\n"
                            buffer = ""
                        elif len(buffer) > 1000:
                            yield f"data: {json.dumps({'type': 'output', 'content': buffer})}\n\n"
                            buffer = ""
                            
                    except Exception as e:
                        break
                
                # 남은 버퍼 전송
                if buffer:
                    yield f"data: {json.dumps({'type': 'output', 'content': buffer})}\n\n"
                
                # 프로세스 완료 대기
                current_process.wait()
                
                # 완료 알림
                if current_process.returncode == 0:
                    yield f"data: {json.dumps({'type': 'complete', 'content': '✅ Anomaly detection completed successfully!'})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'content': f'❌ Anomaly detection failed with exit code: {current_process.returncode}'})}\n\n"
                
                # 프로세스 정리
                with process_lock:
                    current_process = None
                
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': f'Error running anomaly detection: {str(e)}'})}\n\n"
                with process_lock:
                    if current_process:
                        try:
                            current_process.terminate()
                        except:
                            pass
                        current_process = None
        
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
        
    except Exception as e:
        return "data: " + json.dumps({'type': 'error', 'content': str(e)}) + "\n\n", 500

# 실시간 명령어 실행 API (스트리밍)
@app.route('/execute_stream', methods=['GET'])
def execute_stream():
    try:
        command = request.args.get('command', '')
        
        if not command:
            return "data: " + json.dumps({'type': 'error', 'content': 'No command provided'}) + "\n\n", 400
        
        # 보안을 위해 특정 명령어만 금지 (rm, cd는 위험할 수 있음)
        command_parts = command.split()
        forbidden_commands = ['rm', 'cd']
        
        if command_parts and command_parts[0] in forbidden_commands:
            return "data: " + json.dumps({'type': 'error', 'content': f'Command not allowed for security reasons. Forbidden: {", ".join(forbidden_commands)}'}) + "\n\n", 403
        
        # Check if --Anomaly flag is present
        has_anomaly = '--Anomaly' in command
        
        # monit 실행용 명령어 (--Anomaly, --towers 제거)
        monit_command = command
        anomaly_towers = None
        run_number = None
        
        # Parse command
        parts = command.split()
        for i, part in enumerate(parts):
            if part == '--towers' and i + 1 < len(parts):
                anomaly_towers = parts[i + 1]
            elif part == '--RunNumber' and i + 1 < len(parts):
                run_number = parts[i + 1]
        
        if has_anomaly:
            # --Anomaly와 --towers 제거
            monit_command = command.replace('--Anomaly', '').strip()
            if anomaly_towers:
                monit_command = monit_command.replace(f'--towers {anomaly_towers}', '').strip()
        
        def generate():
            nonlocal run_number, anomaly_towers
            
            # 환경변수가 초기화되지 않았다면 초기화 시도
            if not ENV_INITIALIZED:
                success, message = parse_envset_sh()
                if not success:
                    yield f"data: {json.dumps({'type': 'error', 'content': f'Environment not initialized: {message}'})}\n\n"
                    return
            
            # Anomaly detection: 백그라운드로 anomaly detection 시작 (LIVE 여부 무관)
            anomaly_thread = None
            if has_anomaly and run_number:
                def run_live_anomaly_detection():
                    """백그라운드에서 anomaly detection 실행"""
                    # 즉시 이전 결과 파일 삭제 (빠른 연속 실행 대비)
                    output_json = os.path.join(BASE_DIR, 'output', f'Run_{run_number}_anomaly_live.json')
                    log_file = os.path.join(BASE_DIR, 'output', f'Run_{run_number}_anomaly_live.log')
                    # Anomaly와 Normal PNG 둘 다 삭제
                    png_anomaly_c = os.path.join(BASE_DIR, 'output', f'Run_{run_number}_anomaly_C.png')
                    png_anomaly_s = os.path.join(BASE_DIR, 'output', f'Run_{run_number}_anomaly_S.png')
                    png_normal_c = os.path.join(BASE_DIR, 'output', f'Run_{run_number}_normal_C.png')
                    png_normal_s = os.path.join(BASE_DIR, 'output', f'Run_{run_number}_normal_S.png')
                    
                    for f in [output_json, log_file, png_anomaly_c, png_anomaly_s, png_normal_c, png_normal_s]:
                        if os.path.exists(f):
                            try:
                                os.remove(f)
                            except:
                                pass
                    
                    # 0.dat 파일이 생성될 때까지 대기
                    # Tower 정보로 MID 찾기 위한 준비
                    import pandas as pd
                    
                    # 최신 anomaly 매핑 (mapping_KEK_anomaly.csv) 사용
                    mapping_path = os.path.join(os.path.dirname(BASE_DIR), 'mapping', 'mapping_KEK_anomaly.csv')
                    if not os.path.exists(mapping_path):
                        return
                    
                    try:
                        mapping_df = pd.read_csv(mapping_path)
                    except:
                        return
                    
                    # Tower 리스트에서 첫 번째 tower의 MID 찾기
                    if anomaly_towers:
                        tower_list = []
                        for tower_item in anomaly_towers.split(','):
                            tower_item = tower_item.strip()
                            tower_name = tower_item.split('-')[0] if '-' in tower_item else tower_item
                            if tower_name and tower_name not in tower_list:
                                tower_list.append(tower_name)
                        towers = tower_list if tower_list else ['T1']
                    else:
                        towers = ['T1']
                    
                    # 첫 번째 tower의 C 채널로 파일 체크
                    first_tower = towers[0]
                    full_tower_name = f"{first_tower}-C"
                    matching_rows = mapping_df[mapping_df['pmt'].astype(str).str.strip() == full_tower_name.strip()]
                    
                    if len(matching_rows) == 0:
                        return
                    
                    mid = int(matching_rows.iloc[0]['mid'])
                    
                    # 0.dat 파일 경로
                    data_dir = "/Volumes/SSD_8TB"  # detect_live.py와 동일
                    target_file_pattern = os.path.join(data_dir, 
                                                      f"Run_{run_number}/Run_{run_number}_Wave/Run_{run_number}_Wave_MID_{mid}/Run_{run_number}_Wave_MID_{mid}_FILE_0.dat")
                    
                    # 파일이 생성되고 충분한 크기가 될 때까지 대기 (최대 60초)
                    max_wait = 60
                    wait_interval = 1
                    elapsed = 0
                    min_file_size = 64 + 32736*2 * 100  # 최소 100 이벤트
                    
                    while elapsed < max_wait:
                        if os.path.exists(target_file_pattern):
                            file_size = os.path.getsize(target_file_pattern)
                            if file_size >= min_file_size:
                                # 파일이 충분히 크면 detection 시작
                                break
                        time.sleep(wait_interval)
                        elapsed += wait_interval
                    
                    # 파일이 없으면 종료
                    if not os.path.exists(target_file_pattern):
                        return
                    
                    # Tower 리스트 결정
                    if anomaly_towers:
                        tower_list = []
                        for tower_item in anomaly_towers.split(','):
                            tower_item = tower_item.strip()
                            tower_name = tower_item.split('-')[0] if '-' in tower_item else tower_item
                            if tower_name and tower_name not in tower_list:
                                tower_list.append(tower_name)
                        towers = tower_list if tower_list else ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']
                    else:
                        towers = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']
                    
                    tower_str = ','.join(towers)
                    
                    # detect_live.py 경로
                    detect_script = os.path.join(os.path.dirname(BASE_DIR), 'anomaly', 'detect_live.py')
                    output_json = os.path.join(BASE_DIR, 'output', f'Run_{run_number}_anomaly_live.json')
                    log_file = os.path.join(BASE_DIR, 'output', f'Run_{run_number}_anomaly_live.log')
                    
                    # Execute 버튼을 누르면 항상 재실행 - 이전 결과 파일 삭제
                    if os.path.exists(output_json):
                        os.remove(output_json)
                    if os.path.exists(log_file):
                        os.remove(log_file)
                    # PNG는 덮어쓰기되므로 삭제 불필요
                    
                    # 항상 Detection 실행
                    if os.path.exists(detect_script):
                        # 모델과 매핑 경로 설정
                        model_dir = os.path.join(os.path.dirname(BASE_DIR), 'anomaly')
                        mapping_file = os.path.join(os.path.dirname(BASE_DIR), 'mapping', 'mapping_KEK_anomaly.csv')
                        
                        python_cmd = f"cd {os.path.dirname(BASE_DIR)}/anomaly && python3 detect_live.py --run {run_number} --towers {tower_str} --output {output_json} --models {model_dir} --mapping {mapping_file}"
                        
                        try:
                            with open(log_file, 'w') as log_f:
                                # Popen으로 프로세스 시작 (추적 가능하도록)
                                anomaly_proc = subprocess.Popen(
                                    python_cmd,
                                    shell=True,
                                    cwd=os.path.dirname(BASE_DIR),
                                    env=CUSTOM_ENV,
                                    stdout=log_f,
                                    stderr=log_f
                                )
                                
                                # 프로세스 추적 리스트에 추가
                                with anomaly_lock:
                                    anomaly_detection_processes[run_number] = anomaly_proc
                                
                                # 프로세스 완료 대기
                                anomaly_proc.wait()
                                
                                # 프로세스 추적 리스트에서 제거
                                with anomaly_lock:
                                    if run_number in anomaly_detection_processes:
                                        del anomaly_detection_processes[run_number]
                            
                            # Detection 결과가 성공적이면 PNG 파일 체크
                            if anomaly_proc.returncode == 0 and os.path.exists(output_json):
                                try:
                                    with open(output_json, 'r') as f:
                                        detection_result = json.load(f)
                                    
                                    # PNG 파일이 생성되었으면 로그에 추가
                                    if detection_result.get('is_anomaly') and 'png_files' in detection_result:
                                        with open(log_file, 'a') as log_f:
                                            log_f.write(f"\n\n🖼️  PNG files generated:\n")
                                            for png_file in detection_result['png_files']:
                                                log_f.write(f"   - {png_file}\n")
                                except:
                                    pass
                        except Exception as e:
                            # 에러 로그 저장
                            try:
                                with open(log_file, 'a') as log_f:
                                    log_f.write(f"\n\nError: {str(e)}\n")
                            except:
                                pass
                            finally:
                                # 에러 발생 시에도 추적 리스트에서 제거
                                with anomaly_lock:
                                    if run_number in anomaly_detection_processes:
                                        del anomaly_detection_processes[run_number]
                
                # 스레드로 실행
                anomaly_thread = threading.Thread(target=run_live_anomaly_detection, daemon=True)
                anomaly_thread.start()
                
                yield f"data: {json.dumps({'type': 'anomaly_started', 'content': f'🔍 Live anomaly detection started for Run {run_number}'})}\n\n"
            
            global current_process
            
            try:
                # Popen을 사용해서 실시간 출력 스트리밍
                import io
                
                with process_lock:
                    current_process = subprocess.Popen(
                        monit_command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,  # stderr를 stdout과 합침
                        text=True,
                        bufsize=1,  # 라인 버퍼링
                        universal_newlines=True,
                        cwd=BASE_DIR,
                        env=CUSTOM_ENV
                    )
                
                # 명령어 시작 알림
                yield f"data: {json.dumps({'type': 'command', 'content': f'$ {command}'})}\n\n"
                
                # 실시간으로 출력 스트리밍 - 개선된 버전
                buffer = ""
                waiting_for_content_after_escape = False
                
                while True:
                    # 프로세스가 종료되었는지 확인
                    if current_process.poll() is not None:
                        break
                    
                    try:
                        # 한 글자씩 읽어서 ANSI 시퀀스 보존
                        char = current_process.stdout.read(1)
                        if not char:
                            continue
                            
                        buffer += char
                        
                        # \x1b[F 패턴을 감지했으면 다음 내용까지 기다림
                        if buffer.endswith('\x1b[F'):
                            waiting_for_content_after_escape = True
                            continue
                        
                        # \x1b[F 후에 내용이 완성되면 함께 전송
                        if waiting_for_content_after_escape and (char == '\n' or char == '\r' or buffer.endswith('\x1b[0m')):
                            yield f"data: {json.dumps({'type': 'output', 'content': buffer.rstrip()})}\n\n"
                            buffer = ""
                            waiting_for_content_after_escape = False
                        # \r로 끝나는 라인도 덮어쓰기 대상으로 처리
                        elif char == '\r' and not waiting_for_content_after_escape:
                            yield f"data: {json.dumps({'type': 'output', 'content': buffer})}\n\n"
                            buffer = ""
                        # 일반 줄바꿈시 전송 (이스케이프 대기 중이 아닐 때만)
                        elif char == '\n' and not waiting_for_content_after_escape:
                            if buffer and buffer.strip():  # 빈 줄은 전송하지 않음
                                # f-string에서 백슬래시 사용 불가로 인한 분리
                                content = buffer.rstrip('\n')
                                yield f"data: {json.dumps({'type': 'output', 'content': content})}\n\n"
                            buffer = ""
                        # 너무 긴 버퍼는 강제로 전송 (무한 대기 방지)
                        elif len(buffer) > 1000:
                            yield f"data: {json.dumps({'type': 'output', 'content': buffer})}\n\n"
                            buffer = ""
                            waiting_for_content_after_escape = False
                            
                    except Exception as e:
                        break
                
                # 남은 버퍼 전송
                if buffer:
                    yield f"data: {json.dumps({'type': 'output', 'content': buffer})}\n\n"
                
                # 프로세스 완료 대기
                current_process.wait()
                
                # 완료 알림
                monit_returncode = current_process.returncode
                yield f"data: {json.dumps({'type': 'complete', 'content': f'DQM plotting exited with code: {monit_returncode}'})}\n\n"
                
                # 프로세스 완료 후 정리
                with process_lock:
                    current_process = None
                
                # Anomaly detection은 백그라운드에서 실행 중 (결과는 왼쪽 창에 표시하지 않음)
                
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': f'Error: {str(e)}'})}\n\n"
                # 에러 발생시에도 프로세스 정리
                with process_lock:
                    if current_process:
                        try:
                            current_process.terminate()
                        except:
                            pass
                        current_process = None
        
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
        
    except Exception as e:
        return "data: " + json.dumps({'type': 'error', 'content': str(e)}) + "\n\n", 500

# 모니터링 프로그램 실행 API
@app.route('/run_monit', methods=['POST'])
def run_monit():
    try:
        # 환경변수가 초기화되지 않았다면 먼저 초기화
        if not ENV_INITIALIZED:
            success, message = parse_envset_sh()
            if not success:
                return jsonify({
                    'command': './monit',
                    'stdout': '',
                    'stderr': f'Environment not initialized: {message}',
                    'returncode': 1
                })
        
        # 라이브러리 의존성 체크
        lib_status = check_library_dependencies()
        
        # monit 프로그램 실행
        result = run_command_with_env('./monit')
        
        # 에러가 발생한 경우 추가 정보 제공
        if result.returncode != 0 and 'Library not loaded' in result.stderr:
            additional_info = f"""

🔍 Library Loading Error Detected!

Library Status Check:
{lib_status}

💡 Troubleshooting Tips:
1. Try building the project again: cd .. && ./buildNinstall.sh
2. Check if ROOT is properly installed: which root
3. Verify library permissions: ls -la install/lib/libdrcTB.dylib

For more help with library loading issues, try running:
otool -L ./monit  (to check dependencies)
"""
            return jsonify({
                'command': './monit',
                'stdout': result.stdout,
                'stderr': result.stderr + additional_info,
                'returncode': result.returncode
            })
        
        return jsonify({
            'command': './monit',
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 환경설정 초기화 API
@app.route('/init_env', methods=['POST'])
def init_env():
    try:
        global ENV_INITIALIZED
        
        # 환경변수 파싱 및 초기화
        success, message = parse_envset_sh()
        
        if success:
            # 환경변수가 제대로 설정되었는지 확인
            env_info = []
            important_vars = ['ROOTSYS', 'DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH', 'LD_LIBRARY_PATH', 'PATH', 'INSTALL_DIR_PATH', 'YAML_CPP_DIR']
            
            for var in important_vars:
                if var in CUSTOM_ENV:
                    env_info.append(f"{var}={CUSTOM_ENV[var]}")
                else:
                    env_info.append(f"{var}=<not set>")
            
            # 라이브러리 의존성 체크
            lib_check = check_library_dependencies()
            
            # rpath 문제 체크 및 자동 수정
            rpath_fix_result = check_and_fix_rpath()
            
            stdout_msg = f"""Environment initialization successful!

Key environment variables:
{chr(10).join(env_info)}

Library Status:
{lib_check}

RPath Fix:
{rpath_fix_result}

libdrcTB.dylib location: {os.path.join(os.path.dirname(BASE_DIR), 'install', 'lib', 'libdrcTB.dylib')}
"""
            
            return jsonify({
                'command': 'Environment Initialization',
                'stdout': stdout_msg,
                'stderr': '',
                'returncode': 0
            })
        else:
            return jsonify({
                'command': 'Environment Initialization',
                'stdout': '',
                'stderr': f'Failed to initialize environment: {message}',
                'returncode': 1
            })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def check_and_fix_rpath():
    """monit 실행파일의 rpath를 체크하고 필요시 수정합니다."""
    try:
        monit_path = os.path.join(BASE_DIR, "monit")
        
        if not os.path.exists(monit_path):
            return "❌ monit executable not found"
        
        # 현재 rpath 확인
        result = subprocess.run(
            ['otool', '-l', monit_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return "⚠️  Could not read rpath information"
        
        # 현재 ROOT 라이브러리 경로 찾기
        current_root_lib = None
        
        # 1. ROOTSYS 환경변수에서 찾기
        if 'ROOTSYS' in CUSTOM_ENV:
            potential_lib = os.path.join(CUSTOM_ENV['ROOTSYS'], 'lib')
            if os.path.exists(potential_lib):
                current_root_lib = potential_lib
        
        # 2. Homebrew Cellar에서 찾기 (백업)
        if not current_root_lib:
            try:
                root_cellar_path = "/opt/homebrew/Cellar/root"
                if os.path.exists(root_cellar_path):
                    versions = [d for d in os.listdir(root_cellar_path) if os.path.isdir(os.path.join(root_cellar_path, d))]
                    if versions:
                        latest_version = sorted(versions)[-1]
                        potential_lib = f"/opt/homebrew/Cellar/root/{latest_version}/lib/root"
                        if os.path.exists(potential_lib):
                            current_root_lib = potential_lib
            except Exception:
                pass
        
        # 3. 표준 경로들 시도
        if not current_root_lib:
            standard_paths = [
                "/opt/homebrew/lib/root",
                "/usr/local/lib/root", 
                "/usr/lib/root"
            ]
            for path in standard_paths:
                if os.path.exists(path):
                    current_root_lib = path
                    break
        
        if not current_root_lib:
            # 더 적극적으로 해결 - DYLD_LIBRARY_PATH에 install/lib 추가
            install_lib_path = os.path.join(os.path.dirname(BASE_DIR), "install", "lib")
            if os.path.exists(install_lib_path):
                return f"⚠️  ROOT library path not found, but install/lib exists.\n✅ Adding install/lib to DYLD_LIBRARY_PATH should resolve the issue.\n   Path: {install_lib_path}"
            else:
                return "❌ Could not determine ROOT library path and install/lib not found"
        
        # rpath에서 ROOT 경로 찾기
        lines = result.stdout.split('\n')
        old_rpath = None
        for i, line in enumerate(lines):
            if 'LC_RPATH' in line and i + 2 < len(lines):
                path_line = lines[i + 2].strip()
                if 'path ' in path_line and 'root' in path_line:
                    old_rpath = path_line.split('path ')[1].split(' (offset')[0]
                    break
        
        # libdrcTB.dylib 위치 확인
        install_lib_path = os.path.join(os.path.dirname(BASE_DIR), "install", "lib")
        libdrc_source = os.path.join(install_lib_path, "libdrcTB.dylib")
        libdrc_target = os.path.join(current_root_lib, "libdrcTB.dylib")
        
        if not os.path.exists(libdrc_source):
            return "❌ libdrcTB.dylib not found in install/lib"
        
        # ROOT 라이브러리 디렉토리에 libdrcTB.dylib가 있는지 확인
        if os.path.exists(libdrc_target):
            if os.path.islink(libdrc_target):
                # 심볼릭 링크인 경우 올바른 경로를 가리키는지 확인
                link_target = os.readlink(libdrc_target)
                if os.path.samefile(libdrc_source, libdrc_target):
                    return f"✅ libdrcTB.dylib symbolic link already exists and correct\n   Link: {libdrc_target} -> {link_target}"
                else:
                    # 잘못된 링크 제거 후 재생성
                    try:
                        os.unlink(libdrc_target)
                    except Exception as e:
                        return f"❌ Failed to remove incorrect symbolic link: {str(e)}"
            else:
                return f"⚠️  libdrcTB.dylib already exists in ROOT lib (not a link): {libdrc_target}"
        
        # rpath 업데이트 시도 (선택사항)
        rpath_result = ""
        if old_rpath and old_rpath != current_root_lib:
            try:
                # 이전 rpath 제거 시도
                subprocess.run(
                    ['install_name_tool', '-delete_rpath', old_rpath, monit_path],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                # 새 rpath 추가 시도
                add_result = subprocess.run(
                    ['install_name_tool', '-add_rpath', current_root_lib, monit_path],
                    capture_output=True,
                    text=True
                )
                
                if add_result.returncode == 0:
                    rpath_result = f"✅ RPath updated: {old_rpath} -> {current_root_lib}\n"
                else:
                    rpath_result = f"⚠️  RPath update failed (will use symlink): {add_result.stderr}\n"
                    
            except Exception as e:
                rpath_result = f"⚠️  RPath update error (will use symlink): {str(e)}\n"
        elif old_rpath == current_root_lib:
            rpath_result = f"✅ RPath already correct: {current_root_lib}\n"
        else:
            rpath_result = "⚠️  No ROOT rpath found, will create symlink\n"
        
        # 심볼릭 링크 생성
        try:
            # sudo 없이 시도 (권한이 있는 경우)
            try:
                os.symlink(libdrc_source, libdrc_target)
                symlink_result = f"✅ Created symbolic link: {libdrc_target} -> {libdrc_source}"
            except PermissionError:
                # sudo 권한이 필요한 경우
                sudo_result = subprocess.run(
                    ['sudo', 'ln', '-sf', libdrc_source, libdrc_target],
                    capture_output=True,
                    text=True,
                    input='\n'  # 빈 입력으로 패스워드 프롬프트 건너뛰기 시도
                )
                
                if sudo_result.returncode == 0:
                    symlink_result = f"✅ Created symbolic link (sudo): {libdrc_target} -> {libdrc_source}"
                else:
                    symlink_result = f"❌ Failed to create symbolic link: {sudo_result.stderr}"
                    
        except Exception as e:
            symlink_result = f"❌ Error creating symbolic link: {str(e)}"
        
        return rpath_result + symlink_result
        
    except Exception as e:
        return f"❌ Error in rpath check/fix: {str(e)}"

# RPath 수정 전용 API
@app.route('/fix_rpath', methods=['POST'])
def fix_rpath():
    try:
        result = check_and_fix_rpath()
        
        return jsonify({
            'command': 'RPath Fix',
            'stdout': result,
            'stderr': '',
            'returncode': 0 if '✅' in result else 1
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/SetENV', methods=['POST'])
def SetENV():
    try:
        # monit 프로그램 실행
        result = run_command_with_env('cd .. ; source envset.sh ; cd monit')
        
        return jsonify({
            'command': 'cd .. ; source envset.sh ; cd monit',
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 프로세스 종료 API
@app.route('/kill_process', methods=['POST'])
def kill_process():
    try:
        global current_process
        
        killed_monit = False
        killed_anomaly = False
        
        # 1. monit 프로세스 종료
        with process_lock:
            if current_process is not None and current_process.poll() is None:
                # 프로세스가 실행 중인 경우 종료
                current_process.terminate()
                # 강제 종료가 필요한 경우
                try:
                    current_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    current_process.kill()
                    current_process.wait()
                
                killed_monit = True
        
        # 2. 백그라운드 anomaly detection 프로세스 종료
        with anomaly_lock:
            if anomaly_detection_processes:
                for run_num, proc in list(anomaly_detection_processes.items()):
                    if proc.poll() is None:  # 실행 중인 경우
                        try:
                            proc.terminate()
                            proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        except:
                            pass
                        
                        # 결과 파일 삭제 (Anomaly와 Normal PNG 둘 다)
                        output_json = os.path.join(BASE_DIR, 'output', f'Run_{run_num}_anomaly_live.json')
                        log_file = os.path.join(BASE_DIR, 'output', f'Run_{run_num}_anomaly_live.log')
                        png_anomaly_c = os.path.join(BASE_DIR, 'output', f'Run_{run_num}_anomaly_C.png')
                        png_anomaly_s = os.path.join(BASE_DIR, 'output', f'Run_{run_num}_anomaly_S.png')
                        png_normal_c = os.path.join(BASE_DIR, 'output', f'Run_{run_num}_normal_C.png')
                        png_normal_s = os.path.join(BASE_DIR, 'output', f'Run_{run_num}_normal_S.png')
                        
                        for f in [output_json, log_file, png_anomaly_c, png_anomaly_s, png_normal_c, png_normal_s]:
                            if os.path.exists(f):
                                try:
                                    os.remove(f)
                                except:
                                    pass
                        
                        killed_anomaly = True
                
                # 모든 프로세스 제거
                anomaly_detection_processes.clear()
        
        # 결과 메시지 생성
        if killed_monit and killed_anomaly:
            message = 'Monit and Anomaly Detection processes terminated successfully'
        elif killed_monit:
            message = 'Monit process terminated successfully'
        elif killed_anomaly:
            message = 'Anomaly Detection processes terminated successfully'
        else:
            return jsonify({
                'success': False,
                'message': 'No running process to kill'
            })
        
        return jsonify({
            'success': True,
            'message': message
        })
                
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/find_monit_processes', methods=['GET'])
def find_monit_processes():
    try:
        import subprocess
        # Find all processes containing './monit'
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({'success': False, 'message': 'Failed to get process list'})
        
        processes = []
        lines = result.stdout.strip().split('\n')
        
        for line in lines[1:]:  # Skip header
            if './monit' in line:
                # Parse process info
                parts = line.split(None, 10)  # Split into max 11 parts
                if len(parts) >= 11:
                    pid = parts[1]
                    command = parts[10]
                    processes.append({
                        'pid': pid,
                        'command': command,
                        'full_line': line
                    })
        
        return jsonify({
            'success': True, 
            'processes': processes,
            'count': len(processes)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/kill_all_monit', methods=['POST'])
def kill_all_monit():
    try:
        import subprocess
        import signal
        
        # Get list of ./monit processes first
        find_result = find_monit_processes()
        if not find_result.get_json()['success']:
            return find_result
        
        processes = find_result.get_json()['processes']
        
        if not processes:
            return jsonify({'success': True, 'message': 'No ./monit processes found', 'killed_count': 0})
        
        killed_pids = []
        failed_pids = []
        
        for proc in processes:
            try:
                pid = int(proc['pid'])
                os.kill(pid, signal.SIGTERM)
                killed_pids.append(pid)
                
                # Give process time to terminate gracefully
                import time
                time.sleep(0.5)
                
                # Check if still running and force kill if necessary
                try:
                    os.kill(pid, 0)  # Check if process exists
                    os.kill(pid, signal.SIGKILL)  # Force kill
                except OSError:
                    pass  # Process already terminated
                
            except (ValueError, OSError) as e:
                failed_pids.append(proc['pid'])
        
        # Kill all anomaly detection processes and remove results
        with anomaly_lock:
            for run_num, anomaly_proc in list(anomaly_detection_processes.items()):
                if anomaly_proc.poll() is None:
                    try:
                        anomaly_proc.terminate()
                        anomaly_proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        anomaly_proc.kill()
                        anomaly_proc.wait()
                    except:
                        pass
                
                # 결과 파일 삭제 (Anomaly와 Normal PNG 둘 다)
                output_json = os.path.join(BASE_DIR, 'output', f'Run_{run_num}_anomaly_live.json')
                log_file = os.path.join(BASE_DIR, 'output', f'Run_{run_num}_anomaly_live.log')
                png_anomaly_c = os.path.join(BASE_DIR, 'output', f'Run_{run_num}_anomaly_C.png')
                png_anomaly_s = os.path.join(BASE_DIR, 'output', f'Run_{run_num}_anomaly_S.png')
                png_normal_c = os.path.join(BASE_DIR, 'output', f'Run_{run_num}_normal_C.png')
                png_normal_s = os.path.join(BASE_DIR, 'output', f'Run_{run_num}_normal_S.png')
                
                for f in [output_json, log_file, png_anomaly_c, png_anomaly_s, png_normal_c, png_normal_s]:
                    if os.path.exists(f):
                        try:
                            os.remove(f)
                        except:
                            pass
            
            anomaly_detection_processes.clear()
        
        message = f"Killed {len(killed_pids)} ./monit process(es)"
        if failed_pids:
            message += f", failed to kill: {failed_pids}"
        
        return jsonify({
            'success': True, 
            'message': message,
            'killed_count': len(killed_pids),
            'failed_count': len(failed_pids),
            'killed_pids': killed_pids,
            'failed_pids': failed_pids
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# 실행 중인 프로세스 상태 확인 API
@app.route('/process_status', methods=['GET'])
def process_status():
    try:
        global current_process
        
        with process_lock:
            if current_process is not None and current_process.poll() is None:
                return jsonify({
                    'running': True,
                    'pid': current_process.pid
                })
            else:
                return jsonify({
                    'running': False
                })
                
    except Exception as e:
        return jsonify({
            'running': False,
            'error': str(e)
        })

# 보안 코드 검증 API
@app.route('/verify_code', methods=['POST'])
def verify_code():
    try:
        data = request.get_json()
        code = data.get('code', '')
        
        # 간단한 보안 코드 (실제 환경에서는 더 안전한 방법 사용)
        SECURITY_CODE = "KEK"
        
        if code == SECURITY_CODE:
            return jsonify({
                'success': True,
                'message': 'Access granted'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid access code'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/system-info')
def get_system_info():
    try:
        import psutil
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return jsonify({
            'memory': {
                'used': round(memory.used / 1024 / 1024 / 1024, 2),  # GB
                'total': round(memory.total / 1024 / 1024 / 1024, 2),  # GB
                'percent': memory.percent
            },
            'cpu': {
                'percent': cpu_percent
            }
        })
    except ImportError:
        # Fallback if psutil is not available
        import os
        import resource
        
        # Get memory usage (rough estimate)
        memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On macOS, ru_maxrss is in bytes; on Linux, it's in KB
        if os.name == 'posix' and os.uname().sysname == 'Darwin':  # macOS
            memory_mb = memory_usage / 1024 / 1024
        else:  # Linux
            memory_mb = memory_usage / 1024
            
        return jsonify({
            'memory': {
                'used': round(memory_mb / 1024, 2),  # GB
                'total': 'N/A',
                'percent': 'N/A'
            },
            'cpu': {
                'percent': 'N/A'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dqm_manual.pdf')
def serve_dqm_manual():
    try:
        pdf_path = os.path.join(os.path.dirname(__file__), 'dqm_manual.pdf')
        if os.path.exists(pdf_path):
            return send_file(pdf_path, mimetype='application/pdf', as_attachment=False)
        else:
            return "DQM Manual PDF not found. Please ensure DRC_DQM_manual.pdf is in the monit directory.", 404
    except Exception as e:
        return f"Error serving DQM manual: {str(e)}", 500

# AI Agent 관리 API
@app.route('/agent/launch', methods=['POST'])
def launch_agent():
    """AI Agent를 실행합니다."""
    try:
        global agent_process
        
        # 현재 요청의 호스트 정보 가져오기 (원격 접속 지원)
        request_host = request.host.split(':')[0]  # 포트 제거
        if request_host == 'localhost' or request_host == '127.0.0.1':
            # localhost인 경우 실제 서버 IP 가져오기
            import socket
            try:
                # 외부 연결을 위한 IP 주소 가져오기
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                request_host = s.getsockname()[0]
                s.close()
            except:
                request_host = request.host.split(':')[0]
        
        with agent_lock:
            # 이미 실행 중인지 확인
            if agent_process is not None and agent_process.poll() is None:
                # 기존 포트 확인 (로그에서 추출 시도)
                agent_port = 5001
                try:
                    import re
                    log_file = os.path.join(os.path.dirname(BASE_DIR), 'logs', 'agent.log')
                    if os.path.exists(log_file):
                        with open(log_file, 'r') as f:
                            log_content = f.read()
                            port_match = re.search(r'로컬 접속:\s*http://[^\s:]+:(\d+)', log_content)
                            if not port_match:
                                port_match = re.search(r'Running on\s+http://[^\s:]+:(\d+)', log_content)
                            if port_match:
                                agent_port = int(port_match.group(1))
                except:
                    pass
                
                return jsonify({
                    'success': True,
                    'message': 'AI Agent is already running',
                    'pid': agent_process.pid,
                    'url': f'http://{request_host}:{agent_port}'
                })
            
            # agent.py 경로
            agent_script = os.path.join(os.path.dirname(BASE_DIR), 'agent.py')
            
            if not os.path.exists(agent_script):
                return jsonify({
                    'success': False,
                    'message': f'Agent script not found: {agent_script}'
                }), 404
            
            # 로그 파일 경로
            log_dir = os.path.join(os.path.dirname(BASE_DIR), 'logs')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, 'agent.log')
            
            # 로그 파일 열기
            log_f = open(log_file, 'w')
            
            # 환경변수 준비 - Flask 관련 환경변수 제거
            env = os.environ.copy()
            # Flask가 파일 디스크립터를 사용하려고 하는 것을 방지
            env.pop('FLASK_RUN_FROM_CLI', None)
            env.pop('WERKZEUG_RUN_MAIN', None)
            env.pop('SERVER_SOFTWARE', None)
            # 디버거 관련 환경변수 제거
            env.pop('FLASK_DEBUG', None)
            env.pop('WERKZEUG_DEBUG_PIN', None)
            
            # agent.py 실행 (stdout/stderr를 로그 파일로)
            agent_process = subprocess.Popen(
                ['python3', agent_script],
                cwd=os.path.dirname(BASE_DIR),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True  # 새로운 세션에서 시작
            )
            
            # 프로세스가 시작될 때까지 대기 및 상태 확인
            import time
            max_wait = 10  # 최대 10초 대기
            waited = 0
            agent_started = False
            
            while waited < max_wait:
                time.sleep(0.5)
                waited += 0.5
                
                # 프로세스가 죽었는지 확인
                if agent_process.poll() is not None:
                    # 프로세스가 종료됨 - 로그 확인
                    log_f.close()
                    with open(log_file, 'r') as f:
                        error_log = f.read()
                    return jsonify({
                        'success': False,
                        'message': 'Agent failed to start',
                        'error': error_log[-500:] if error_log else 'Unknown error'
                    }), 500
                
                # Agent가 실제로 시작되었는지 포트로 확인
                # 로그 파일에서 포트 번호 추출 시도
                agent_port = None
                try:
                    import re
                    if os.path.exists(log_file):
                        with open(log_file, 'r') as f:
                            log_content = f.read()
                            # 여러 패턴 시도
                            # 1. "로컬 접속: http://localhost:5002" 패턴
                            port_match1 = re.search(r'로컬 접속:\s*http://[^\s:]+:(\d+)', log_content)
                            if port_match1:
                                agent_port = int(port_match1.group(1))
                            else:
                                # 2. "Running on http://127.0.0.1:5002" 패턴 (첫 번째)
                                port_match2 = re.search(r'Running on\s+http://[^\s:]+:(\d+)', log_content)
                                if port_match2:
                                    agent_port = int(port_match2.group(1))
                            print(f"DEBUG: Extracted port from log: {agent_port}")
                except Exception as e:
                    print(f"DEBUG: Error extracting port: {e}")
                    pass
                
                # 포트를 찾았으면 해당 포트부터 확인
                if agent_port:
                    try:
                        import socket
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(0.5)
                        result = sock.connect_ex(('localhost', agent_port))
                        sock.close()
                        if result == 0:
                            agent_started = True
                            break
                    except:
                        pass
                else:
                    # 포트를 찾지 못했으면 여러 포트 확인 (5001부터 5010까지)
                    for check_port in range(5001, 5011):
                        try:
                            import socket
                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            sock.settimeout(0.5)
                            result = sock.connect_ex(('localhost', check_port))
                            sock.close()
                            if result == 0:
                                agent_started = True
                                agent_port = check_port
                                break
                        except:
                            continue
                
                if agent_started and agent_port:
                    break
            
            if not agent_started:
                # 타임아웃 - 프로세스는 살아있지만 포트가 안 열림
                return jsonify({
                    'success': False,
                    'message': 'Agent process started but no port responding',
                    'hint': 'Check logs/agent.log for details'
                }), 500
            
            return jsonify({
                'success': True,
                'message': 'AI Agent started successfully',
                'pid': agent_process.pid,
                'url': f'http://{request_host}:{agent_port}'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error launching agent: {str(e)}'
        }), 500

@app.route('/agent/status', methods=['GET'])
def agent_status():
    """AI Agent의 실행 상태를 확인합니다."""
    try:
        global agent_process
        
        # 현재 요청의 호스트 정보 가져오기 (원격 접속 지원)
        request_host = request.host.split(':')[0]  # 포트 제거
        if request_host == 'localhost' or request_host == '127.0.0.1':
            # localhost인 경우 실제 서버 IP 가져오기
            import socket
            try:
                # 외부 연결을 위한 IP 주소 가져오기
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                request_host = s.getsockname()[0]
                s.close()
            except:
                request_host = request.host.split(':')[0]
        
        with agent_lock:
            if agent_process is not None and agent_process.poll() is None:
                # 포트 확인 (로그에서 추출 시도)
                agent_port = 5001
                try:
                    import re
                    log_file = os.path.join(os.path.dirname(BASE_DIR), 'logs', 'agent.log')
                    if os.path.exists(log_file):
                        with open(log_file, 'r') as f:
                            log_content = f.read()
                            port_match = re.search(r'로컬 접속:\s*http://[^\s:]+:(\d+)', log_content)
                            if not port_match:
                                port_match = re.search(r'Running on\s+http://[^\s:]+:(\d+)', log_content)
                            if port_match:
                                agent_port = int(port_match.group(1))
                except:
                    pass
                
                return jsonify({
                    'running': True,
                    'pid': agent_process.pid,
                    'url': f'http://{request_host}:{agent_port}'
                })
            else:
                return jsonify({
                    'running': False
                })
                
    except Exception as e:
        return jsonify({
            'running': False,
            'error': str(e)
        })

@app.route('/agent/stop', methods=['POST'])
def stop_agent():
    """AI Agent를 종료합니다."""
    try:
        global agent_process
        
        with agent_lock:
            if agent_process is not None and agent_process.poll() is None:
                try:
                    # 프로세스 종료
                    agent_process.terminate()
                    try:
                        agent_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        agent_process.kill()
                        agent_process.wait()
                except Exception as e:
                    # 프로세스가 이미 종료되었을 수 있음
                    pass
                
                agent_process = None
                return jsonify({
                    'success': True,
                    'message': 'AI Agent stopped successfully'
                })
            else:
                # 이미 종료된 상태
                agent_process = None
                return jsonify({
                    'success': True,
                    'message': 'AI Agent was not running'
                })
                
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error stopping agent: {str(e)}'
        }), 500

# Anomaly Detection 결과 조회 API
@app.route('/anomaly_status/<int:run_number>', methods=['GET'])
def get_anomaly_status(run_number):
    """Live anomaly detection 결과 조회 (PNG 파일 포함)"""
    try:
        # BASE_DIR의 output 폴더에서 찾음
        output_json = os.path.join(BASE_DIR, 'output', f'Run_{run_number}_anomaly_live.json')
        
        if os.path.exists(output_json):
            with open(output_json, 'r') as f:
                result = json.load(f)
            
            # PNG 파일이 실제로 존재하는지 확인
            if 'png_files' in result:
                verified_png_files = []
                for png_file in result['png_files']:
                    png_path = os.path.join(BASE_DIR, 'output', png_file)
                    if os.path.exists(png_path):
                        verified_png_files.append(png_file)
                result['png_files'] = verified_png_files
            
            return jsonify(result)
        else:
            # 로그 파일도 확인해서 에러가 있는지 체크
            log_file = os.path.join(BASE_DIR, 'output', f'Run_{run_number}_anomaly_live.log')
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    if log_content.strip():
                        return jsonify({'status': 'running', 'message': 'Detection in progress', 'log_preview': log_content[-500:]})
            
            return jsonify({'status': 'pending', 'message': 'Detection in progress or not started'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print(f"✅ Server starting...")
    print(f"📁 Serving ROOT files from: {ROOT_DIR}")
    print(f"🌐 Serving web files from: {INDEX_DIR}")
    print(f"💻 Command execution enabled (forbidden: rm, cd)")
    
    # 서버 시작 시 자동으로 환경 초기화 및 라이브러리 설정
    print(f"🔧 Initializing environment and fixing library paths...")
    
    # 1. 환경변수 초기화
    success, message = parse_envset_sh()
    if success:
        print(f"✅ Environment initialized: {message}")
        
        # 중요한 라이브러리 경로 정보 출력
        install_lib_path = os.path.join(os.path.dirname(BASE_DIR), "install", "lib")
        libdrc_path = os.path.join(install_lib_path, "libdrcTB.dylib")
        
        print(f"📍 Key paths:")
        print(f"   Install lib: {install_lib_path}")
        print(f"   libdrcTB.dylib: {'✅ Found' if os.path.exists(libdrc_path) else '❌ Missing'}")
        
        if 'DYLD_LIBRARY_PATH' in CUSTOM_ENV:
            print(f"   DYLD_LIBRARY_PATH: {CUSTOM_ENV['DYLD_LIBRARY_PATH'][:100]}...")
        if 'ROOTSYS' in CUSTOM_ENV:
            print(f"   ROOTSYS: {CUSTOM_ENV['ROOTSYS']}")
    else:
        print(f"⚠️  Environment initialization warning: {message}")
    
    # 2. 라이브러리 의존성 확인
    try:
        lib_checks = check_library_dependencies()
        print(f"🔍 Library dependency checks:")
        # lib_checks는 문자열이므로 줄바꿈으로 분리해서 출력
        for line in lib_checks.split('\n'):
            if line.strip():
                print(f"   {line}")
    except Exception as e:
        print(f"⚠️  Library dependency check failed: {str(e)}")
    
    # 3. rpath 자동 수정
    try:
        rpath_result = check_and_fix_rpath()
        print(f"🔧 RPath fix result:")
        for line in rpath_result.split('\n'):
            if line.strip():
                print(f"   {line}")
    except Exception as e:
        print(f"⚠️  RPath fix failed: {str(e)}")
    
    print(f"✅ Server ready on http://localhost:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)
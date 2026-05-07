#!/usr/bin/env python3
"""
Configuration Loader
config_general.yml 파일에서 설정을 읽어오는 유틸리티
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_ROOT / "config_general.yml"

# 설정 캐시
_config_cache: Optional[Dict[str, Any]] = None


def load_config() -> Dict[str, Any]:
    """
    config_general.yml 파일을 읽어서 딕셔너리로 반환
    
    Returns:
        설정 딕셔너리
    """
    global _config_cache
    
    if _config_cache is not None:
        return _config_cache
    
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"설정 파일을 찾을 수 없습니다: {CONFIG_FILE}\n"
            f"config_general.yml 파일이 존재하는지 확인하세요."
        )
    
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            _config_cache = yaml.safe_load(f) or {}
        return _config_cache
    except Exception as e:
        raise RuntimeError(f"설정 파일 읽기 실패: {e}")


def get_data_directory() -> str:
    """
    데이터 디렉토리 경로 반환 (BaseDirectory 사용)
    
    Returns:
        데이터 디렉토리 경로
    """
    config = load_config()
    base_dir = config.get("BaseDirectory")
    if not base_dir:
        raise ValueError(
            "설정 파일에 'BaseDirectory'가 없습니다.\n"
            f"config_general.yml 파일에 다음을 추가하세요:\n"
            f"  BaseDirectory: \"/path/to/data\""
        )
    return str(base_dir)


def get_mapping_csv_path() -> str:
    """
    매핑 CSV 파일 경로 반환 (Mapping 경로에서 .root를 .csv로 변환)
    
    Returns:
        매핑 CSV 파일 경로
    """
    config = load_config()
    mapping = config.get("Mapping")
    if not mapping:
        raise ValueError(
            "설정 파일에 'Mapping'이 없습니다.\n"
            f"config_general.yml 파일에 다음을 추가하세요:\n"
            f"  Mapping: \"./mapping/mapping_KEK.root\""
        )
    
    # Mapping 경로가 상대 경로면 프로젝트 루트 기준으로 절대 경로로 변환
    if not os.path.isabs(mapping):
        # DQM 디렉토리 기준으로 변환 (하위 호환성)
        dqm_dir = PROJECT_ROOT / "DQM"
        mapping_path = (dqm_dir / mapping).resolve()
    else:
        mapping_path = Path(mapping)
    
    # .root를 .csv로 변환
    mapping_csv_path = mapping_path.parent / (mapping_path.stem + ".csv")
    
    return str(mapping_csv_path)


def get_trainset_csv_path() -> Optional[str]:
    """
    Trainset CSV 파일 경로 반환 (선택적, hv_equalization_tool에서만 사용)
    
    Returns:
        Trainset CSV 파일 경로 (없으면 None)
    """
    config = load_config()
    trainset_csv = config.get("TrainsetCSV")
    if trainset_csv:
        return str(trainset_csv)
    return None


def get_base_directory() -> str:
    """
    Base 디렉토리 경로 반환 (sample_data)
    
    Returns:
        Base 디렉토리 경로
    """
    config = load_config()
    base_dir = config.get("BaseDirectory")
    if not base_dir:
        raise ValueError(
            "설정 파일에 'BaseDirectory'가 없습니다.\n"
            f"config_general.yml 파일에 다음을 추가하세요:\n"
            f"  BaseDirectory: \"/path/to/sample_data\""
        )
    return str(base_dir)


def get_path_config(key: str) -> str:
    """
    설정 파일의 Paths 섹션에서 경로를 가져옴
    
    Args:
        key: Paths 섹션의 키 (RunNumberFile, JsonKeyFile 등)
        
    Returns:
        설정된 경로 문자열
    """
    config = load_config()
    paths = config.get("Paths", {})
    val = paths.get(key)
    if not val:
        raise ValueError(f"설정 파일의 Paths 섹션에 '{key}'가 정의되지 않았습니다.")
    
    # SpreadsheetId는 경로가 아니므로 변환 제외
    if key == "SpreadsheetId":
        return str(val)
        
    # 상대 경로인 경우 프로젝트 루트와 결합하여 절대 경로로 변환
    if not os.path.isabs(str(val)):
        return str((PROJECT_ROOT / str(val)).resolve())
        
    return str(val)


def get_hv_config() -> Dict[str, Any]:
    """HV 설정 반환"""
    config = load_config()
    return config.get("HV", {})


def reload_config():
    """설정 캐시를 초기화하여 다시 로드"""
    global _config_cache
    _config_cache = None
    return load_config()

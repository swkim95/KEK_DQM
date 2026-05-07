#!/usr/bin/env python3
"""
Base Agent Class
모든 Sub-Agent의 기본 클래스
"""

import json
import torch
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MAX_CONVERSATION_HISTORY, MAX_NEW_TOKENS


class BaseAgent(ABC):
    """
    모든 Agent의 기본 클래스
    
    Features:
    - State 관리
    - Conversation history 관리 (최근 N개)
    - LLM 로드/언로드 (메모리 효율)
    - Context 생성
    """
    
    def __init__(self, model_path: str, agent_name: str, io_handler=None):
        """
        Args:
            model_path: Fine-tuned 모델 경로
            agent_name: Agent 이름 (로깅용)
            io_handler: IOHandler instance (None → TerminalIO)
        """
        self.model_path = Path(model_path)
        self.agent_name = agent_name

        # IO handler (TerminalIO by default keeps existing CLI behavior)
        if io_handler is None:
            from agents.io_handler import TerminalIO
            self.io = TerminalIO()
        else:
            self.io = io_handler
        
        # 모델 (필요할 때 로드)
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # State (각 Agent가 override)
        self.state = {}
        
        # Conversation History
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history = MAX_CONVERSATION_HISTORY
    
    # ===== Context Manager (자동 로드/언로드) =====
    
    def __enter__(self):
        """with 문 시작: 모델 로드"""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """with 문 종료: 모델 언로드"""
        self.unload()
        return False
    
    # ===== 모델 관리 =====
    
    def load(self):
        """모델 로드"""
        if self.model is not None:
            return  # 이미 로드됨
        
        # Device 설정
        if torch.backends.mps.is_available():
            self.device = "mps"
            print(f"  ✅ [{self.agent_name}] MPS 사용")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print(f"  ✅ [{self.agent_name}] CUDA 사용")
        else:
            self.device = "cpu"
            print(f"  ✅ [{self.agent_name}] CPU 사용")
        
        # Tokenizer 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=True
        )
        
        print(f"  ✅ [{self.agent_name}] 모델 로드 완료: {self.model_path}")
    
    def unload(self):
        """모델 언로드 (메모리 해제)"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        print(f"  🗑️  [{self.agent_name}] 모델 언로드 완료")
    
    # ===== 대화 관리 =====
    
    def add_to_history(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        대화 히스토리에 추가
        
        Args:
            role: "user" or "assistant"
            content: 대화 내용
            metadata: 추가 메타데이터 (선택)
        """
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            entry.update(metadata)
        
        self.conversation_history.append(entry)
        
        # 최대 개수 제한
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_recent_history(self, n: Optional[int] = None) -> List[Dict]:
        """최근 N개 대화 반환"""
        if n is None:
            n = self.max_history
        return self.conversation_history[-n:]
    
    # ===== Context 생성 =====
    
    @abstractmethod
    def _build_state_context(self) -> str:
        """
        State를 문자열로 변환 (각 Agent가 구현)
        
        Returns:
            State 정보를 담은 문자열
        """
        pass
    
    def _build_history_context(self) -> str:
        """대화 히스토리를 문자열로 변환 (최근 대화 우선)"""
        if not self.conversation_history:
            return "(No conversation yet)"
        
        lines = []
        # 최근 대화를 먼저 보여주기 (가장 최근 10개)
        recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Agent"
            content = msg["content"]
            
            # JSON인 경우 메시지만 추출하거나 간략화
            if role == "Agent":
                try:
                    decision = json.loads(content)
                    if "message" in decision:
                        content = decision["message"]
                    elif "tool" in decision:
                        content = f"[Tool Call: {decision['tool']}]"
                        if "update_state" in decision:
                            content += f" (Update State: {list(decision['update_state'].keys())})"
                except:
                    pass

            # "완료"가 있으면 명확하게 표시
            if content == "완료" or content.strip() == "완료":
                lines.append(f"{role}: 완료 [IMPORTANT: User confirmed completion]")
            else:
                lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def build_full_context(self, current_input: Optional[str] = None) -> str:
        """
        전체 context 생성
        
        Args:
            current_input: 현재 사용자 입력
        
        Returns:
            State + History + Current Input을 합친 context
        """
        parts = []
        
        # 1. State
        parts.append("=== Current State ===")
        parts.append(self._build_state_context())
        parts.append("")
        
        # 2. History
        parts.append("=== Recent Conversation ===")
        parts.append(self._build_history_context())
        parts.append("")
        
        # 3. Current Input
        if current_input:
            parts.append("=== Current User Input ===")
            parts.append(current_input)
            parts.append("")
        
        # 4. Task
        parts.append("=== Your Task ===")
        parts.append("Based on the current state and conversation, decide the next action.")
        parts.append("Output JSON with tool name and parameters.")
        
        return "\n".join(parts)
    
    # ===== LLM 호출 =====
    
    def decide(self, context: str) -> Dict[str, Any]:
        """
        LLM에게 다음 행동 결정 요청
        
        Args:
            context: Full context string (현재 단계의 context)
        
        Returns:
            {"tool": "tool_name", "params": {...}, "reason": "..."}
        """
        if self.model is None:
            raise RuntimeError(f"[{self.agent_name}] Model not loaded. Use with statement or call load().")
        
        # System prompt (각 Agent마다 다름)
        system_prompt = self._get_system_prompt()
        
        # Messages 구성 (기본적으로는 현재 context만 보내지만, 
        # fine-tuning 데이터가 multi-turn인 경우 이전 대화 내용을 포함하는 것이 좋음)
        # 하지만 context 자체에 history가 포함되어 있으므로, 
        # fine-tuning 시의 turn 구조를 맞추기 위해 이전 context들을 보낼 수도 있음.
        # 여기서는 가장 최근의 context만 전달하되, 필요하다면 전체 history를 개별 메시지로 전달하도록 변경 가능.
        
        # 현재는 context가 이미 history를 포함하고 있으므로, 단일 turn으로 전달하되
        # fine-tuning 데이터의 형식을 따름.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]
        
        # Chat template 적용
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,          # Greedy decoding (deterministic)
                repetition_penalty=1.1,   # !!! 반복 방지
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # JSON 파싱 (첫 번째 완전한 JSON만 추출)
        try:
            # JSON 추출: 첫 번째 완전한 JSON 객체만 추출
            json_str = generated_text.strip()
            if '{' not in json_str:
                raise json.JSONDecodeError("No JSON found", json_str, 0)
            
            # 첫 번째 '{'부터 시작하여 중괄호 매칭으로 완전한 JSON 추출
            start = json_str.index('{')
            depth = 0
            end = start
            
            for i in range(start, len(json_str)):
                char = json_str[i]
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            
            if depth != 0:
                # 완전한 JSON을 찾지 못함
                raise json.JSONDecodeError("Incomplete JSON", json_str, end)
            
            json_str = json_str[start:end]
            
            # Python boolean → JSON boolean 변환
            json_str = json_str.replace('True', 'true').replace('False', 'false').replace('None', 'null')
            
            # JSON 파싱
            decision = json.loads(json_str)
            return decision
        
        except (json.JSONDecodeError, ValueError) as e:
            return {
                "error": f"JSON parsing failed: {str(e)}",
                "raw_output": generated_text
            }
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """
        System prompt 반환 (각 Agent가 구현)
        """
        pass
    
    # ===== 로깅 =====
    # 파일 저장 없이 stdout 으로만 — `logs/` 디렉터리는 더 이상 만들지 않음.
    # 호출부 (`self.log(...)` 48여 곳)는 그대로 유지되며, 파이프/리다이렉션으로
    # 필요할 때 외부에서 잡아 쓸 수 있다.

    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{self.agent_name}] {message}", flush=True)
    
    # ===== 추상 메서드 =====
    
    @abstractmethod
    def run(self):
        """
        Agent 실행 (각 Agent가 구현)
        """
        pass

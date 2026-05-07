#!/usr/bin/env python3
"""
IO Handler
----------
Abstraction layer between agents and I/O.
Terminal mode keeps existing behavior; WebSocketIO routes through queues.
"""

import queue
import threading
from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path


class IOHandler(ABC):
    @abstractmethod
    def get_input(self) -> str:
        """Block until user provides input. Returns stripped string."""
        pass

    @abstractmethod
    def send_ai_message(self, message: str):
        """Send AI conversation message → left panel."""
        pass

    @abstractmethod
    def send_tool_output(self, text: str):
        """Send tool stdout → right panel."""
        pass

    @abstractmethod
    def send_plots(self, file_paths: List[str]):
        """Send plot image paths → right panel."""
        pass

    @abstractmethod
    def send_status(self, text: str):
        """Update status bar."""
        pass


class TerminalIO(IOHandler):
    """Default: same behavior as before (print / input)."""

    def get_input(self) -> str:
        return input("👤 You: ").strip()

    def send_ai_message(self, message: str):
        print(f"\n💬 Agent: {message}")

    def send_tool_output(self, text: str):
        print(text, flush=True)

    def send_plots(self, file_paths: List[str]):
        for f in file_paths:
            print(f"   📊 Plot saved → {Path(f).name}")

    def send_status(self, text: str):
        print(f"[Status] {text}")


class WebSocketIO(IOHandler):
    """WebSocket-backed I/O for the web UI."""

    # Messages containing these keywords require a physical action from the
    # operator → show the 완료 button.
    # Everything else (value requests) just needs typed input → no button.
    _CONFIRM_KEYWORDS = [
        "이동해주세요",       # move stage / tower
        "설정해주세요",       # set beam energy
        "확인해주세요",       # check results
        "적용하시겠습니까",   # apply HV change
        "다음 DAQ를 시작합니다",  # ready for next DAQ
        "전압이 변경되었습니다",  # HV voltage changed
    ]

    def __init__(self, input_queue: queue.Queue, output_queue: queue.Queue,
                 stop_event: Optional[threading.Event] = None,
                 waiting_flag: Optional[threading.Event] = None):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.waiting_flag = waiting_flag      # set while blocked on get_input()
        self._last_needs_confirm = False   # set by send_ai_message

    def send_ai_message(self, message: str):
        self._last_needs_confirm = any(kw in message for kw in self._CONFIRM_KEYWORDS)
        self.output_queue.put({"type": "ai_message", "content": message})

    def get_input(self) -> str:
        from agents.agent_runner import StopAgentException
        # Only show 완료 button when the last message required a physical action
        if self._last_needs_confirm:
            self.output_queue.put({"type": "awaiting_input"})
        self._last_needs_confirm = False   # reset for next call
        # Signal that the scenario agent is blocked waiting for user input
        if self.waiting_flag is not None:
            self.waiting_flag.set()
        try:
            # Poll so we can react to stop_event even while blocked
            while True:
                if self.stop_event and self.stop_event.is_set():
                    raise StopAgentException()
                try:
                    value = self.input_queue.get(timeout=0.2)
                    return value.strip()
                except queue.Empty:
                    continue
        finally:
            if self.waiting_flag is not None:
                self.waiting_flag.clear()

    def send_tool_output(self, text: str):
        if text and text.strip():
            self.output_queue.put({"type": "tool_output", "content": text})

    def send_plots(self, file_paths: List[str]):
        for path in file_paths:
            # Send just the filename; server serves /plots/<filename>
            self.output_queue.put({"type": "plot", "filename": Path(path).name})

    def send_status(self, text: str):
        self.output_queue.put({"type": "status", "content": text})

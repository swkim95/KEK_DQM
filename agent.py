#!/usr/bin/env python3

import os
os.environ['INTEGRAL_WEB_MODE'] = 'true'

import re
import json
import logging
import uuid
from typing import TypedDict
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

from llm_provider import get_llm
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from tools.mcp_tavily.web_search_server import search_web_tool
from tools.hvcontrol import hv_execute_tool, hv_confirm_command_tool
from tools.hv_equalization import (
    hveq_center_init_tool,
    hveq_session_start_tool,
    hveq_analyze_suggest_tool,
    hveq_add_training_data,
    hveq_system_status,
    hveq_done_channel,
)

# 로깅 설정
for logger_name in ["urllib3", "langsmith"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

llm = get_llm()

# 전역 변수
active_requests = {}  # 진행 중인 요청 추적 {session_id: should_stop}

# 사용 가능한 도구들 (웹 모드 전용)
tools = [
    search_web_tool,
    hv_execute_tool,
    hv_confirm_command_tool,
    hveq_center_init_tool,
    hveq_session_start_tool,
    hveq_analyze_suggest_tool,
    hveq_add_training_data,
    hveq_system_status,
    hveq_done_channel,
]

# 상태 정의
class State(TypedDict):
    messages: list
    context: str
    selected_tools: list
    session_id: str

def create_initial_state(session_id: str = 'default') -> State:
    return {"messages": [], "context": "", "selected_tools": [], "session_id": session_id}

def parse_actions(content):
    """Action과 Action Input 파싱"""
    lines = content.split('\n')
    actions = []
    current_action = None
    current_input = ""
    
    for line in lines:
        line = line.strip()
        if line.startswith('Action:'):
            if current_action:
                actions.append((current_action, current_input))
            current_action = line.split('Action:')[1].strip()
            current_input = ""
        elif line.startswith('Action Input:'):
            current_input = line.split('Action Input:')[1].strip()
    
    if current_action:
        actions.append((current_action, current_input))
    
    return actions

def process_tool_input(tool_name, tool_input):
    """도구별 입력 파라미터 처리"""
    if not tool_input or tool_input.strip() == "":
        if tool_name == "search_web_tool":
            return "ERROR:Please specify what you want to know about"
    
    # JSON에서 문자열 추출 (기본 도구용)
    if tool_name == "search_web_tool":
        try:
            parsed = json.loads(tool_input)
            if isinstance(parsed, dict):
                return parsed.get("question") or parsed.get("query") or parsed.get("topic") or str(parsed)
        except:
            pass
        return tool_input
    
    # HV 도구 파라미터 처리
    elif tool_name in ["hv_execute_tool"]:
        # 자연어 그대로 전달 (AI가 파싱함)
        return tool_input
    
    elif tool_name == "hv_confirm_command_tool":
        # hvctl 명령어 그대로 전달
        return tool_input
    
    return tool_input

def find_tool_by_name(tool_name):
    """이름으로 도구 찾기"""
    for tool in tools:
        if tool.name == tool_name:
            return tool
    return None

def execute_tool(tool, tool_input):
    """도구 실행"""
    # hv_confirm_command_tool의 경우 특별 처리
    if tool.name == "hv_confirm_command_tool":
        if not tool_input or tool_input.strip() == "":
            return "ERROR: No command to execute"
        return tool.invoke(tool_input)
    
    # 다른 도구들의 경우
    if not tool_input or tool_input.strip() == "":
        return tool.invoke({})
    
    try:
        tool_args = json.loads(tool_input)
    except json.JSONDecodeError:
        tool_args = tool_input
    
    if isinstance(tool_args, dict):
        return tool.invoke(tool_args)
    else:
        return tool.invoke(tool_args)

def replace_action_with_result(content, action, input_val, result, is_error=False):
    """Action을 결과로 교체"""
    action_pattern = f"Action: {action}\nAction Input: {input_val or ''}"
    replacement = f"{action_pattern}\nObservation: {result}"
    return content.replace(action_pattern, replacement)

def is_tool_allowed(tool_name: str, selected_tools: list) -> bool:
    """선택된 도구에 따라 도구 사용 허용 여부 확인"""
    always_allowed = ['search_web_tool', 'hv_confirm_command_tool']
    if tool_name in always_allowed:
        return True
    
    tool_categories = {
        'hv': ['hv_execute_tool', 'hv_confirm_command_tool'],
        'hveq': [
            'hveq_center_init_tool',
            'hveq_session_start_tool',
            'hveq_analyze_suggest_tool',
            'hveq_add_training_data',
            'hveq_system_status',
            'hveq_done_channel'
        ]
    }
    
    for category, category_tools in tool_categories.items():
        if tool_name in category_tools:
            return category in selected_tools
    
    return False

def generate_system_prompt(selected_tools):
    """모드별 맞춤 시스템 프롬프트 생성"""
    
    selected_mode = selected_tools[0] if selected_tools else None
    
    if not selected_mode:
        return """
You are an AI assistant for CERN's dual-readout calorimeter test-beam experiments.

GENERAL MODE (No specific tools selected):
- Use RAG search for experiment-related questions
- Use web search for general physics questions  
- Provide informative answers based on retrieved information
- If both searches fail, use your general knowledge

TOOL FORMAT:
Action: tool_name
Action Input: parameters
Observation: (result)
Final Answer: (summary)
"""
    
    tool_info = []
    
    base_prompt = f"""
You are an AI assistant for CERN's dual-readout calorimeter test-beam experiments.

CURRENT MODE: {selected_mode.upper()}

CRITICAL RULES:
1. ONLY use tools from the selected mode
2. Extract parameters from user input automatically
3. For information tools: provide Final Answer after Observation
4. For execution tools: show Observation only

TOOL FORMAT:
Action: tool_name
Action Input: parameters
Observation: (result)
[Final Answer: (for information tools only)]
"""
    
    # 선택된 모드에 따른 전용 설명
    if selected_mode == 'hv':
        tool_info.append("""
HV MODE:
- Available tools: hv_execute_tool
- HV input is automatically processed by AI
- Commands: "slot 2 ch 1 V0Set 10", "채널 3 켜줘", "slot 1 ch 5 turn off"
- Supports CAEN HV parameters: V0Set (voltage) and Pw On/Off
- All HV commands require confirmation for safety
""")
    
    elif selected_mode == 'hveq':
        tool_info.append("""
HV EQUALIZATION MODE:
- Available tools: hveq_center_init_tool, hveq_session_start_tool, hveq_analyze_suggest_tool
- HV Equalization workflow: Center → Target → Run → Analyze → Suggest
- Commands: "center T1", "target C=100 S=150", "run 12345", "run 12345 분석"
- Provides current ADC values and HV adjustment suggestions
- Pure advisory mode - no direct HV control
""")
    
    return base_prompt + "".join(tool_info)

def handle_hv_input(user_input: str, selected_tools: list) -> str:
    """HV 도구가 선택되었을 때 사용자 입력을 자동으로 처리"""
    if 'hv' not in selected_tools:
        return user_input
    
    user_lower = user_input.lower().strip()
    
    # HV 관련 키워드가 있으면 hv_execute_tool 사용
    hv_keywords = ['슬롯', 'slot', '채널', 'channel', 'ch', '전압', 'voltage', 'v0set',
                   '켜', 'on', '꺼', 'off', 'pw']
    
    # HV 관련 키워드가 있거나 숫자가 포함되어 있으면 HV 명령으로 간주
    if (any(keyword in user_lower for keyword in hv_keywords) or 
        re.search(r'\d+', user_input)):
        return f"Action: hv_execute_tool\nAction Input: {user_input}"
    
    # 기본적으로 hv_execute_tool로 처리 (HV 모드에서는)
    return f"Action: hv_execute_tool\nAction Input: {user_input}"

def handle_hveq_input(user_input: str, selected_tools: list) -> str:
    """HV Equalization 도구가 선택되었을 때 사용자 입력을 자동으로 처리"""
    if 'hveq' not in selected_tools:
        return user_input
    
    user_lower = user_input.lower().strip()
    
    # "center" 입력 → center 초기화
    if 'center' in user_lower:
        return f"Action: hveq_center_init_tool\nAction Input: {user_input}"
    
    # "target", "시작" → session start
    if any(word in user_lower for word in ['target', '시작', 'start']):
        return f"Action: hveq_session_start_tool\nAction Input: {user_input}"
    
    # "run" + 숫자 → analyze & suggest (HV 값 선택적)
    if 'run' in user_lower and re.search(r'\d+', user_input):
        return f"Action: hveq_analyze_suggest_tool\nAction Input: {user_input}"
    
    # "분석", "제안" → analyze & suggest (마지막 run 사용)
    if any(word in user_lower for word in ['분석', '제안', 'analyze', 'suggest']):
        return f"Action: hveq_analyze_suggest_tool\nAction Input: {user_input}"
    
    # "data", "add", "training" → add training data
    if any(word in user_lower for word in ['data', 'add', 'training', '데이터', '추가', '학습']):
        return f"Action: hveq_add_training_data\nAction Input: {user_input}"
    
    # "done", "끝", "완료" → mark channel done
    if any(word in user_lower for word in ['done', '끝', '완료', 'finish', 'complete']):
        return f"Action: hveq_done_channel\nAction Input: {user_input}"
    
    # "status", "상태" → system status
    if any(word in user_lower for word in ['status', '상태', 'system', '시스템']):
        return f"Action: hveq_system_status\nAction Input: {user_input}"
    
    # 기본적으로 일반 대화로 처리
    return user_input

def handle_sequential_search(user_input: str, context: str, session_id: str = 'default') -> str:
    """기본 모드에서 RAG → Web → AI 순차 검색"""
    
    print(f"DEBUG: handle_sequential_search called with input: '{user_input}' for session {session_id}")
    
    # 빈 입력이나 무효한 입력 체크
    if not user_input or not user_input.strip():
        print("DEBUG: Empty or invalid input, skipping sequential search")
        return "안녕하세요! 궁금한 것이 있으시면 언제든 물어보세요."
    
    # 중단 체크 함수
    def is_aborted():
        return active_requests.get(session_id, False)
    
    # 1단계: Web 검색 (Tavily)
    try:
        if is_aborted():
            print(f"DEBUG: Request aborted before web search for session {session_id}")
            return "🛑 요청이 중단되었습니다."
            
        print("DEBUG: Starting web search...")
        web_tool = find_tool_by_name("search_web_tool")
        if web_tool:
            web_result = web_tool.invoke(user_input)
            
            if is_aborted():
                print(f"DEBUG: Request aborted after web search for session {session_id}")
                return "🛑 요청이 중단되었습니다."
            print(f"DEBUG: Web result: {str(web_result)[:100]}...")
            
            # Web 검색 실패 감지
            web_failed = any(phrase in str(web_result).lower() for phrase in [
                "no results found", "error", "failed", "unable to search", "search failed"
            ]) or len(str(web_result).strip()) < 10
            
            if not web_failed:
                print("DEBUG: Web search succeeded, returning result")
                # Web 검색 성공 - Final Answer 생성
                system_prompt = """You are a helpful physics AI assistant. Based on the web search results, give a clear and accurate answer."""
                final_prompt = f"{system_prompt}\n\nContext: {context}\n\nUser: {user_input}\n\nWeb Search Results: {web_result}\n\nProvide a Final Answer based on the search results:"
                final_response = llm.invoke([HumanMessage(content=final_prompt)])
                final_content = final_response.content
                
                if "Final Answer:" in final_content:
                    final_answer = final_content.split("Final Answer:")[-1].strip()
                else:
                    final_answer = final_content.strip()
                
                return f"Action: search_web_tool\nAction Input: {user_input}\n\nFinal Answer: {final_answer}"
            else:
                print("DEBUG: Web search failed, moving to AI direct answer")
    except Exception as e:
        print(f"웹 검색 실패: {e}")
    
    # 2단계: AI 직접 답변
    try:
        if is_aborted():
            print(f"DEBUG: Request aborted before AI direct answer for session {session_id}")
            return "🛑 요청이 중단되었습니다."
            
        print("DEBUG: Starting AI direct answer...")
        system_prompt = """You are a helpful physics AI assistant. Use your knowledge to answer the user's question as accurately as possible."""
        direct_prompt = f"{system_prompt}\n\nContext: {context}\n\nUser: {user_input}\n\nProvide a helpful answer based on your knowledge:"
        direct_response = llm.invoke([HumanMessage(content=direct_prompt)])
        
        if is_aborted():
            print(f"DEBUG: Request aborted after AI direct answer for session {session_id}")
            return "🛑 요청이 중단되었습니다."
            
        return direct_response.content
    except Exception as e:
        if is_aborted():
            return "🛑 요청이 중단되었습니다."
        return f"죄송합니다. 답변을 생성하는 중 오류가 발생했습니다: {str(e)}"

def agent_node(state: State) -> State:
    """메인 에이전트 노드"""
    messages = state["messages"]
    context = state.get("context", "")
    selected_tools = state.get("selected_tools", [])
    session_id = state.get("session_id", "default")
    
    last_message = messages[-1] if messages else None
    if not last_message or not isinstance(last_message, HumanMessage):
        return state
    
    # 선택된 모드에 따른 자동 처리 (단일 모드 선택)
    auto_handled = False
    selected_mode = selected_tools[0] if selected_tools else None
    
    # 각 모드별 전용 처리
    if selected_mode == 'hveq':
        hveq_response = handle_hveq_input(last_message.content, selected_tools)
        if hveq_response != last_message.content:
            content = hveq_response
            auto_handled = True
    
    elif selected_mode == 'hv':
        hv_response = handle_hv_input(last_message.content, selected_tools)
        if hv_response != last_message.content:
            content = hv_response
            auto_handled = True
    
    # 자동 처리되지 않은 경우 순차 검색 또는 LLM 호출
    if not auto_handled:
        # 기본 모드(도구 선택 없음)일 때 순차 검색: Tavily Web → AI
        if not selected_tools or selected_tools == []:
            content = handle_sequential_search(last_message.content, context, session_id)
        else:
            # 도구가 선택된 경우 기존 방식
            system_prompt = generate_system_prompt(selected_tools)
            full_prompt = f"{system_prompt}\n\nContext: {context}\n\nUser: {last_message.content}"
            response = llm.invoke([HumanMessage(content=full_prompt)])
            content = response.content
    
    # 도구 호출 처리
    if "Action:" in content:
        actions = parse_actions(content)
        
        for action_name, action_input in actions:
            try:
                # 도구 사용 권한 확인
                if not is_tool_allowed(action_name, selected_tools):
                    content = replace_action_with_result(content, action_name, action_input, 
                        "Tool not available. Select appropriate tool first.", is_error=True)
                    continue
                
                # 도구 찾기
                tool = find_tool_by_name(action_name)
                if not tool:
                    content = replace_action_with_result(content, action_name, action_input, 
                        f"Tool '{action_name}' not found.", is_error=True)
                    continue
                
                # 파라미터 처리
                processed_input = process_tool_input(action_name, action_input)
                if processed_input.startswith("ERROR:"):
                    content = replace_action_with_result(content, action_name, action_input, 
                        processed_input[6:], is_error=True)
                    continue
                
                # 도구 실행
                result = execute_tool(tool, processed_input)
                
                # 특별 응답 처리
                if isinstance(result, str) and (result.startswith("CONFIRM_COMMAND:") or result.startswith("WEB_SESSION:")):
                    new_context = f"{context}\nUser: {last_message.content}\nAI: {result}" if context else f"User: {last_message.content}\nAI: {result}"
                    return {"messages": messages + [AIMessage(content=result)], "context": new_context, "selected_tools": selected_tools, "session_id": session_id}
                
                # 에러 결과 처리 - ERROR로 시작하는 결과는 그대로 전달
                if isinstance(result, str) and result.startswith("ERROR:"):
                    content = replace_action_with_result(content, action_name, action_input, result)
                else:
                    # 일반 결과 처리
                    content = replace_action_with_result(content, action_name, action_input, str(result))
                
            except Exception as e:
                content = replace_action_with_result(content, action_name, action_input, 
                    f"Error executing tool: {str(e)}", is_error=True)
    
    # Final Answer 생성 (정보 도구용)
    if "Observation:" in content and "search_web_tool" in content:
        # Final Answer 생성
        if "Final Answer:" not in content:
            final_prompt = f"{system_prompt}\n\nContext: {context}\n\nUser: {last_message.content}\n\n{content}\n\nProvide a Final Answer based on the Observation:"
            final_response = llm.invoke([HumanMessage(content=final_prompt)])
            final_content = final_response.content
            
            if "Final Answer:" in final_content:
                final_answer = final_content.split("Final Answer:")[-1].strip()
            else:
                final_answer = final_content.strip()
            
            content += f"\n\nFinal Answer: {final_answer}"
        
        # search_web_tool의 경우 Observation 숨기기
        if "search_web_tool" in content:
            # Action: search_web_tool과 Action Input: 은 유지하고 Observation: 부분만 제거
            lines = content.split('\n')
            filtered_lines = []
            skip_observation = False
            
            for line in lines:
                if line.startswith('Observation:'):
                    skip_observation = True
                    continue
                elif line.startswith('Final Answer:'):
                    skip_observation = False
                    filtered_lines.append(line)
                elif not skip_observation:
                    filtered_lines.append(line)
                # skip_observation이 True인 상태에서 Final Answer가 아닌 모든 줄은 스킵
            
            content = '\n'.join(filtered_lines)
    
    # 메시지 추가 및 컨텍스트 업데이트
    messages.append(AIMessage(content=content))
    new_context = f"{context}\nUser: {last_message.content}\nAI: {content}" if context else f"User: {last_message.content}\nAI: {content}"
    
    return {"messages": messages, "context": new_context, "selected_tools": selected_tools, "session_id": session_id}

# 워크플로우 설정
workflow = StateGraph(State)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.set_finish_point("agent")

memory = MemorySaver()
graph_app = workflow.compile(checkpointer=memory)

# Flask 웹 앱 생성
flask_app = Flask(__name__)
CORS(flask_app)

# 세션 관리
conversation_history = {}
session_states = {}

def get_conversation_history(session_id: str = "default"):
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    return conversation_history[session_id]

def get_session_state(session_id: str = "default"):
    if session_id not in session_states:
        session_states[session_id] = {"pending_command": None}
    return session_states[session_id]

def clear_conversation_history(session_id: str = "default"):
    if session_id in conversation_history:
        conversation_history[session_id] = []
    
    if session_id in session_states:
        session_states[session_id] = {"pending_command": None, "pending_integral_feedback": None}
    
    try:
        config = {"configurable": {"thread_id": session_id}}
        graph_app.update_state(config, {"messages": [], "context": "", "selected_tools": []})
    except Exception:
        pass
    
    return True

def run_agent(query: str, session_id: str = "default", selected_tools: list = None) -> str:
    if selected_tools is None:
        selected_tools = []
        
    try:
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            current_state = graph_app.get_state(config)
            if current_state and current_state.values.get("messages"):
                existing_messages = current_state.values["messages"]
                existing_messages.append(HumanMessage(content=query))
                input_data = {
                    "messages": existing_messages,
                    "context": current_state.values.get("context", ""),
                    "selected_tools": selected_tools,
                    "session_id": session_id
                }
            else:
                input_data = {
                    "messages": [HumanMessage(content=query)],
                    "context": "",
                    "selected_tools": selected_tools,
                    "session_id": session_id
                }
        except:
            input_data = {
                "messages": [HumanMessage(content=query)],
                "context": "",
                "selected_tools": selected_tools,
                "session_id": session_id
            }
        
        result = graph_app.invoke(input_data, config=config)
        messages = result["messages"]
        
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                content = message.content
                
                # 시스템 프롬프트 제거
                if "<|system|>" in content:
                    if "<|assistant|>" in content:
                        content = content.split("<|assistant|>")[-1]
                    else:
                        content = content.split("<|end|>")[-1] if "<|end|>" in content else content
                
                return content.strip()
        
        return "Unable to generate response."
        
    except Exception as e:
        return f"Error occurred while running agent: {str(e)}"

# === Flask 웹 라우트 ===

@flask_app.route('/')
def index():
    return render_template('index.html')

@flask_app.route('/api/chat', methods=['POST'])
def chat():
    session_id = 'default'  # 기본값 설정
    try:
        data = request.json
        user_input = data.get('message', '')
        session_id = data.get('session_id', 'default')
        selected_tools = data.get('selected_tools', [])
        
        # 요청 시작 등록
        active_requests[session_id] = False
        print(f"DEBUG: Started request for session {session_id}")
        
        # 에이전트 실행
        response_content = run_agent(user_input, session_id, selected_tools)
        
        # 요청 완료 후 제거
        if session_id in active_requests:
            del active_requests[session_id]
            print(f"DEBUG: Completed request for session {session_id}")
        
        # 특별 응답 처리
        if response_content.startswith('CONFIRM_COMMAND:'):
            return handle_command_confirmation(response_content, session_id)
        
        # 일반 응답
        return jsonify({
            'response': response_content,
            'session_id': session_id
        })
        
    except Exception as e:
        # 요청 에러 시에도 제거
        if session_id in active_requests:
            del active_requests[session_id]
            print(f"DEBUG: Error cleanup for session {session_id}")
        
        return jsonify({
            'error': f'서버 오류가 발생했습니다: {str(e)}',
            'session_id': session_id
        }), 500

def handle_command_confirmation(response_content, session_id):
    """명령 확인 응답 처리"""
    print(f"DEBUG handle_command_confirmation: response_content = '{response_content}'")
    
    # CONFIRM_COMMAND: 다음의 모든 내용을 추출
    if ':' in response_content:
        command_data = ':'.join(response_content.split(':')[1:]).strip()
    else:
        command_data = response_content.replace('CONFIRM_COMMAND', '').strip()
    
    print(f"DEBUG handle_command_confirmation: extracted command_data = '{command_data}'")
    
    return jsonify({
        'response': '다음 명령을 실행하시겠습니까?',
        'session_id': session_id,
        'special_response': 'confirm_command',
        'command_data': command_data
    })

@flask_app.route('/api/session/new', methods=['POST'])
def create_new_session():
    try:
        data = request.json or {}
        old_session_id = data.get('current_session_id')
        
        # 이전 세션 정리
        if old_session_id:
            clear_conversation_history(old_session_id)
        
        # 새 세션 생성
        new_session_id = str(uuid.uuid4())
        clear_conversation_history(new_session_id)
        
        return jsonify({
            'session_id': new_session_id,
            'message': '새로운 대화가 시작되었습니다.'
        })
        
    except Exception as e:
        return jsonify({
            'error': f'새 세션 생성 중 오류가 발생했습니다: {str(e)}'
        }), 500

@flask_app.route('/api/confirm_command', methods=['POST'])
def confirm_command():
    session_id = 'default'  # 기본값 설정
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        confirmed = data.get('confirmed', False)
        command_data = data.get('command_data', '')
        
        if not confirmed:
            return jsonify({
                'response': '명령이 취소되었습니다.',
                'session_id': session_id
            })
        
        # 명령 실행
        try:
            print(f"DEBUG confirm_command: command_data = '{command_data}'")
            print(f"DEBUG confirm_command: confirmed = {confirmed}")
            
            # HV 명령만 지원
            tool_name = "hv_confirm_command_tool"
            
            # 해당 도구 찾기
            execute_tool_obj = None
            for tool in tools:
                if tool.name == tool_name:
                    execute_tool_obj = tool
                    break
            
            if not execute_tool_obj:
                return jsonify({
                    'response': f'ERROR: {tool_name} tool not found',
                    'session_id': session_id
                })
            
            # 도구 실행
            print(f"DEBUG confirm_command: About to execute {tool_name} with command_data: '{command_data}'")
            result = execute_tool(execute_tool_obj, command_data)
            print(f"DEBUG confirm_command: Result: '{result}'")
            
            return jsonify({
                'response': result,
                'session_id': session_id
            })
            
        except Exception as e:
            return jsonify({
                'response': f'명령 실행 중 오류가 발생했습니다: {str(e)}',
                'session_id': session_id
            })
        
    except Exception as e:
        return jsonify({
            'error': f'명령 확인 중 오류가 발생했습니다: {str(e)}',
            'session_id': session_id
        }), 500

@flask_app.route('/api/abort', methods=['POST'])
def abort_request():
    """일반 요청 중단"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id:
            active_requests[session_id] = True  # 중단 플래그 설정
            print(f"DEBUG: Abort request set for session {session_id}")
            
            return jsonify({
                'response': '🛑 요청이 중단되었습니다.',
                'success': True
            })
        else:
            return jsonify({
                'response': '세션 ID가 없습니다.',
                'success': False
            }), 400
            
    except Exception as e:
        print(f"DEBUG: Exception aborting request: {e}")
        return jsonify({
            'response': f'요청 중단 실패: {str(e)}',
            'success': False
        }), 500

@flask_app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    print("🚀 웹 서버를 시작합니다...")
    # IP 주소 확인
    def get_local_ip():
        try:
            import socket
            # 외부 서버와 연결 시도하여 로컬 IP 확인
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect(("8.8.8.8", 80))
            local_ip = sock.getsockname()[0]
            sock.close()
            return local_ip
        except:
            return "localhost"
    
    local_ip = get_local_ip()
    
    # 포트가 사용 중인지 확인하고 사용 가능한 포트 찾기
    def find_free_port(start_port=5001, max_attempts=10):
        """사용 가능한 포트를 찾습니다."""
        import socket
        for i in range(max_attempts):
            port = start_port + i
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        raise RuntimeError(f"Could not find a free port starting from {start_port}")
    
    port = find_free_port(5001)
    
    print("=" * 60)
    print("🚀 AI Physicist 서버가 시작되었습니다!")
    print("=" * 60)
    print(f"📱 로컬 접속: http://localhost:{port}")
    print(f"🌐 네트워크 접속: http://{local_ip}:{port}")
    print("=" * 60)
    
    # 필수 디렉토리 생성
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Flask 관련 환경변수 정리 (subprocess 실행 시 문제 방지)
    os.environ.pop('FLASK_RUN_FROM_CLI', None)
    os.environ.pop('WERKZEUG_RUN_MAIN', None)
    os.environ.pop('SERVER_SOFTWARE', None)
    
    flask_app.run(
        debug=False,  # subprocess에서는 debug=False로
        host='0.0.0.0', 
        port=port, 
        use_reloader=False,
        threaded=True,
        use_debugger=False
    )

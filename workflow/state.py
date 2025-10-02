# =============================================================================
# ALL IMPORTS - ORGANIZED BY CATEGORY
# =============================================================================

# Standard Library
import os
import json
import csv
from datetime import datetime
from collections import Counter

# Third-party Libraries
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

# LangChain Core
from langchain_core.documents import Document
from langchain_core.messages import (
    BaseMessage, 
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    ToolMessage
)
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

# LangChain Upstage
from langchain_upstage import ChatUpstage

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

# Pydantic
from pydantic import BaseModel, ValidationError

# Typing
from typing import (
    List, 
    Dict, 
    Any, 
    TypedDict, 
    Annotated, 
    Sequence, 
    Literal
)
from typing import Optional

# Load environment variables
load_dotenv(verbose=True)



class AgentState(TypedDict):
    """The state of the agent."""
    input_code: Annotated[str, "Input_Code"] # 사용자 입력 코드
    initial_analysis: Annotated[str, "Local LLM이 생성한 초기 취약점 분석 결과 (텍스트)"]# 위험 요소 메시지(Local LLM output)

    #retrieved_vulnerabilities: List[Dict[str, Any]] # 취약점 2차원 리스트 (RAG output)
    #matched_vulnerabilities: List[str] # 위험 정도 2차원 리스트 (RAG output)
    retrieved_vulnerabilities: Annotated[List[Dict[str, Any]], "Vector DB에서 검색된 관련 취약점 정보 목록"]

    matched_vulnerabilities: Annotated[List[str], "입력 코드와 최종적으로 매칭된 취약점 이름 목록"]  # 최종 매칭 취약점 리스트
    final_severity: Annotated[str, "가장 높은 위험도로 결정된 최종 심각도 (0 ~ 10)"] # 최종 매칭 위험 정도
    fixed_code: Annotated[str, "취약점이 수정된 추천 코드"] # 수정된 코드
    report: Annotated[str, "사용자에게 보여줄 최종 분석 보고서"] # 출력할 레포트
    is_detected: Annotated[bool, "취약점 존재 여부"]



class ResponseData(BaseModel):
    fixed_code: Optional[str] = ""
    changelog: Optional[List[str]] = []
    notes: Optional[str] = None


model = ChatUpstage(model="solar-pro2", temperature=0)
# tools = [search_knowledge_base, web_search]
# model = model.bind_tools(tools)


# 로컬 LLM이 취약점을 찾았다면 RAG로, 없다면 바로 보고서 작성하는 분기 함수
def detection_branch(state: AgentState) -> Literal["rag_node", "report_generation_node"]:

    print("--- DETECTION CHECK ---")

    raw = state.get("is_detected", None)

    def _truthy(x: Any) -> bool:
        if isinstance(x, bool):
            return x
        if x is None:
            return False
        if isinstance(x, (int, float)):
            return bool(x)
        s = str(x).strip().lower()
        if s in {"true", "1", "yes", "y", "t"}:
            return True
        if s in {"false", "0", "no", "n", "f", ""}:
            return False
        return False

    try:
        detected = _truthy(raw)
        print(f"is_detected raw: {raw} -> interpreted: {detected}")
    except Exception as e:
        print(f"Error interpreting is_detected: {e}")
        detected = False

    if not detected:
        mv = state.get("matched_vulnerabilities", [])
        if isinstance(mv, list) and len(mv) > 0:
            print("No positive is_detected flag but matched_vulnerabilities is non-empty -> routing to [rag_node]")
            return "rag_node"
        else:
            print("No vulnerabilities detected -> routing to [report_generation_node]")
            return "report_generation_node"

    # detected == True
    print("Vulnerability detected -> routing to [rag_node]")
    return "rag_node"




# final_severity 점수를 확인하여 다음 단계를 결정하는 분기 노드
def severity_branch(state: AgentState) -> Literal["vulnerability_fix_node", "report_generation_node"]:
    
    print("--- SEVERITY CHECK ---")
    severity_str = state.get("final_severity")

    # final_severity 값이 state에 없는 경우를 대비한 방어 코드
    if severity_str is None:
        print("Severity not found, defaulting to report generation.")
        return "report_generation_node"
    
    try:
        # 문자열 형태의 점수를 정수로 변환
        severity_score = int(severity_str)
        print(f"Severity score: {severity_score}")

        if 7 <= severity_score <= 10:
            print("High severity detected -> Routing to [vulnerability_fix_node]")
            return "vulnerability_fix_node"
        else:
            print("Low/Medium severity detected -> Routing to [report_generation_node]")
            return "report_generation_node"
            
    except (ValueError, TypeError):
        # final_severity가 숫자로 변환될 수 없는 경우 (e.g., "High", "N/A")
        print(f"Invalid severity format: '{severity_str}'. Defaulting to report generation.")
        return "report_generation_node"




# 외부 모델에 개선된 코드 받아오는 노드
def vulnerability_fix_node(state: AgentState) -> Dict[str, Any]:


    input_code = state.get("input_code", "") or ""
    matched = state.get("matched_vulnerabilities", []) or []

    # System / User prompt 구성: JSON_only 형식 강제
    system = SystemMessage(
        "You are a secure-code assistant. Given a list of detected vulnerabilities and the original source code, "
        "produce a VALID JSON object ONLY (no extra commentary) with these keys:\n"
        "  - fixed_code: string with the full corrected source code (empty string if no change).\n"
        "  - changelog: array of short strings describing each change and which vulnerability it addresses.\n"
        "  - notes: optional string with additional recommendations.\n\n"
        "Return strictly valid JSON (no markdown, no surrounding backticks)."
    )

    vuln_text = "None" if not matched else "\n".join(f"- {v}" for v in matched)
    user_content = (
        f"Detected vulnerabilities:\n{vuln_text}\n\n"
        "Original code (below):\n"
        f"{input_code}\n\n"
        "Return the JSON ONLY (no extra commentary). Ensure JSON is parseable."
    )
    user_msg = HumanMessage(user_content)

    raw_text = ""
    parsed, validated, errors = {}, None, []

    # 모델 호출 & 파싱
    try:
        resp = model.invoke([system, user_msg])
        raw_text = resp.content if hasattr(resp, "content") else str(resp)
        parsed = json.loads(raw_text) if raw_text.strip().startswith("{") else print('error')
    except Exception as e:
        errors.append(str(e))

    # Pydantic 검증 시도
    try:
        norm = {
            "fixed_code": parsed.get("fixed_code", ""),
            "changelog": parsed.get("changelog", []),
            "notes": parsed.get("notes"),
        }
        if isinstance(norm["changelog"], str):
            norm["changelog"] = [ln.strip("-• ") for ln in norm["changelog"].splitlines() if ln.strip()]
        validated = ResponseData(**norm)
    except Exception as ve:
        errors.append(f"Validation error: {ve}")

    # === state 업데이트용 dict 리턴 ===
    return {
        "fixed_code": validated.fixed_code if validated else parsed.get("fixed_code", ""),
        "report": f"### 수정된 코드\n\n{validated.fixed_code if validated else ''}\n\n"
                  f"### 변경 사항\n{validated.changelog if validated else ''}\n\n"
                  f"### 추가 노트\n{validated.notes if validated else ''}",
        # "messages": [raw_text],
        # "errors": errors,
    }



  # 보고서 작성하는 노드 

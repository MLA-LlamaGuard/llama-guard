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


# All node implementations and branch functions have been moved to nodes.py 

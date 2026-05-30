#!/usr/bin/env python3
"""
graph.py

LangGraph workflow construction and execution for LlamaGuard vulnerability analysis.
"""

import os
import sys
import argparse
import sqlite3
import path_setup  # noqa: F401 — adds project root and llama-model/ to sys.path
from langgraph.graph import StateGraph, END
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    _USE_SQLITE = True
except ImportError:
    from langgraph.checkpoint.memory import InMemorySaver
    _USE_SQLITE = False

from state import AgentState
from CVE.cve_vectordb import CVEEntry

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Make CVEEntry available in __main__ namespace for pickle compatibility
# (needed when pickle file was created from a script run as __main__)
sys.modules['__main__'].CVEEntry = CVEEntry
from nodes import (
    initial_analysis_node,
    rag_node,
    cvss_calculation_node,
    report_generation_node,
    detection_branch,
)

# SQLite checkpoint DB path: always in project root, regardless of CWD
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH = os.path.join(_PROJECT_DIR, "llamaguard_state.db")


def build_graph():
    """
    Build the LangGraph StateGraph for vulnerability analysis workflow.

    Workflow:
        START
          ↓
        initial_analysis_node
          ↓
        detection_branch
          ↓ (is_detected?)
          ├─ False → report_generation_node → END
          └─ True → rag_node
                      ↓
                    cvss_calculation_node
                      ↓
                    report_generation_node → END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("initial_analysis_node", initial_analysis_node)
    workflow.add_node("rag_node", rag_node)
    workflow.add_node("cvss_calculation_node", cvss_calculation_node)
    workflow.add_node("report_generation_node", report_generation_node)

    workflow.set_entry_point("initial_analysis_node")

    workflow.add_conditional_edges(
        "initial_analysis_node",
        detection_branch,
        {
            "rag_node": "rag_node",
            "report_generation_node": "report_generation_node",
        }
    )

    workflow.add_edge("rag_node", "cvss_calculation_node")
    workflow.add_edge("cvss_calculation_node", "report_generation_node")
    workflow.add_edge("report_generation_node", END)

    if _USE_SQLITE:
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        memory = SqliteSaver(conn)
    else:
        memory = InMemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph


def run_analysis(input_code: str, thread_id: str = "default"):
    """
    Run vulnerability analysis on the provided code.

    Args:
        input_code: Source code to analyze
        thread_id: Thread ID for checkpointing (default: "default")

    Returns:
        Final state dictionary with all accumulated fields
    """
    print("\n" + "=" * 80)
    print("LlamaGuard Vulnerability Analysis")
    print("=" * 80)

    graph = build_graph()
    initial_state = {"input_code": input_code}
    run_config = {"configurable": {"thread_id": thread_id}}

    # stream(mode="updates") yields per-node deltas — used only for progress output.
    for update in graph.stream(initial_state, run_config):
        for node_name in update:
            print(f"\n[{node_name}] completed")

    # Retrieve full accumulated state (all fields from every node).
    try:
        snapshot = graph.get_state(run_config)
        final_state = snapshot.values if snapshot else {}
    except Exception as e:
        print(f"WARNING: Could not retrieve final state: {e}")
        final_state = {}

    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)

    return final_state


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="LlamaGuard Vulnerability Analysis Workflow")
    parser.add_argument("--code", type=str, help="Code to analyze (direct input)")
    parser.add_argument("--code_file", type=str, help="Path to code file to analyze")
    parser.add_argument("--output", type=str, default=None, help="Path to save report")
    args = parser.parse_args()

    # Get input code
    if args.code:
        code = args.code
        print("[Input] Direct code input")
    elif args.code_file:
        with open(args.code_file, "r", encoding="utf-8") as f:
            code = f.read()
        print(f"[Input] Code from {args.code_file}")
    else:
        # Default example
        code = (
            "def login(username, password):\n"
            "    query = f\"SELECT * FROM users WHERE username='{username}' AND password='{password}'\"\n"
            "    cursor.execute(query)\n"
            "    return cursor.fetchone()\n"
        )
        print("[Input] Using default example (SQL Injection)")

    # Run analysis
    final_state = run_analysis(code)

    # Print report
    if final_state and "report" in final_state:
        print("\n" + "=" * 80)
        print("FINAL REPORT")
        print("=" * 80)
        print(final_state["report"])

        # Save to file if requested
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(final_state["report"])
            print(f"\nReport saved to: {args.output}")
    else:
        print("\nERROR: No report generated")


if __name__ == "__main__":
    main()

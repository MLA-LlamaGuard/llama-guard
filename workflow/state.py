# =============================================================================
# STATE DEFINITIONS FOR LANGGRAPH WORKFLOW
# =============================================================================

from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any, Annotated

# Load environment variables
load_dotenv()


class AgentState(TypedDict):
    """
    State of the vulnerability analysis workflow.

    Fields:
        input_code: User-provided source code to analyze
        initial_analysis: LLaMA model vulnerability analysis output
        retrieved_vulnerabilities: Similar CVEs from vector database (RAG)
        matched_vulnerabilities: Extracted vulnerability type names
        final_severity: CVSS score (0-10) averaged from related CVEs
        report: Final analysis report for user
        is_detected: Whether vulnerabilities were detected
    """
    input_code: Annotated[str, "User input code"]
    initial_analysis: Annotated[str, "LLaMA vulnerability analysis"]
    retrieved_vulnerabilities: Annotated[List[Dict[str, Any]], "Related CVEs from vector DB"]
    matched_vulnerabilities: Annotated[List[str], "Vulnerability type names"]
    final_severity: Annotated[str, "CVSS score (0-10)"]
    report: Annotated[str, "Final report"]
    is_detected: Annotated[bool, "Vulnerability detected flag"]

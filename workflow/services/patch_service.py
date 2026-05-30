#!/usr/bin/env python3
"""
patch_service.py

Vulnerability processing module for LlamaGuard.

Handles external LLM API calls to generate security patches and reports for vulnerable code.
Uses configuration from config.py for all thresholds and API settings.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional

# Add parent directory to path for config import
_svc_dir = os.path.dirname(os.path.abspath(__file__))
_workflow_dir = os.path.dirname(_svc_dir)
if _workflow_dir not in sys.path:
    sys.path.insert(0, _workflow_dir)

from config import config

# ---------------------------
# External LLM call (OpenAI-compatible client)
# ---------------------------
def call_external_for_patch(vuln: str, code: str, language: str) -> Dict[str, Any]:
    """
    Call Upstage Solar API to generate security patch.

    Args:
        vuln: Vulnerability type/name
        code: Original vulnerable code
        language: Programming language

    Returns:
        {"vuln": ..., "patched_code": {"language": ..., "code_snippet": ...}}

    Raises:
        RuntimeError: If UPSTAGE_API_KEY not set or API call fails
    """
    if not config.UPSTAGE_API_KEY:
        raise RuntimeError('UPSTAGE_API_KEY environment variable is required')

    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError('OpenAI SDK not installed. Install with: pip install openai>=1.52.2')

    client = OpenAI(api_key=config.UPSTAGE_API_KEY, base_url=config.UPSTAGE_BASE_URL)

    system_prompt = (
        "You are a senior security engineer and code reviewer. Given vulnerable code and vuln metadata, "
        "produce a JSON object EXACTLY matching the schema: {\"vuln\":..., \"patched_code\":{\"language\":...,\"code_snippet\":...}}. "
        "The code_snippet should be the FULL corrected/patched version of the original code. "
        "Do NOT include any extra explanatory text, do not output secrets, file paths, or PoC exploits. "
        "Return ONLY valid JSON (no markdown, no backticks)."
    )

    user_prompt = (
        f"vuln: {vuln}\n"
        f"language: {language}\n\n"
        f"vulnerable_code:\n{code}\n\n"
        "Return a single JSON object with the patched code (no extra commentary)."
    )

    try:
        resp = client.chat.completions.create(
            model=config.UPSTAGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=config.UPSTAGE_TEMPERATURE,
            max_tokens=config.UPSTAGE_MAX_TOKENS,
            stream=False
        )
    except Exception as ex:
        raise RuntimeError(f'API call failed: {ex}')

    # Extract content
    try:
        content = resp.choices[0].message.content.strip()
    except (AttributeError, IndexError) as e:
        raise RuntimeError(f'Unexpected API response format: {e}')

    # Parse JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code block
        import re
        match = re.search(r'\{[\s\S]*\}', content)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise RuntimeError(f'Could not parse JSON from API response: {content[:200]}...')

# ---------------------------
# Complete security report generation
# ---------------------------
def generate_security_report(
    vuln: str,
    code: str,
    language: str,
    cvss_score: float,
    llama_analysis: str,
    related_cves: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Call Upstage Solar API to generate complete security report with patch.

    Args:
        vuln: Vulnerability type/name (e.g., "SQL Injection")
        code: Original vulnerable code
        language: Programming language
        cvss_score: CVSS severity score (0-10)
        llama_analysis: LLaMA model's vulnerability analysis
        related_cves: Retrieved CVE evidence from RAG search

    Returns:
        {
            "vuln": "SQL Injection",
            "cvss_score": 9.8,
            "executive_summary": "This is a critical severity vulnerability...",
            "potential_impact": [
                "Attackers can read, modify, or delete database records",
                "Potential for privilege escalation...",
                ...
            ],
            "attack_difficulty": "Low",
            "required_privileges": "None",
            "recommended_mitigation": {
                "immediate": "Use parameterized queries for all database operations",
                "short_term": "Implement input validation and sanitization",
                "long_term": "Deploy WAF and conduct security audit"
            },
            "implementation_steps": [
                "Replace all dynamic SQL queries with prepared statements",
                "Use ORM frameworks or database-specific parameterized query APIs",
                ...
            ],
            "patched_code": {
                "language": "python",
                "code_snippet": "def login(...):\\n    query = ..."
            },
            "estimated_effort_hours": 6,
            "confidence": 0.85
        }

    Raises:
        RuntimeError: If UPSTAGE_API_KEY not set or API call fails
    """
    if not config.UPSTAGE_API_KEY:
        raise RuntimeError('UPSTAGE_API_KEY environment variable is required')

    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError('OpenAI SDK not installed. Install with: pip install openai>=1.52.2')

    client = OpenAI(api_key=config.UPSTAGE_API_KEY, base_url=config.UPSTAGE_BASE_URL)
    cve_context = "No related CVEs were provided."
    if related_cves:
        cve_lines = []
        for cve in related_cves:
            cve_lines.append(
                "- "
                f"{cve.get('cve_id', 'N/A')} | "
                f"CVSS: {cve.get('cvss', 'N/A')} | "
                f"CWE: {cve.get('cwe', 'N/A')} | "
                f"Similarity: {cve.get('similarity', 0):.4f}\n"
                f"  Evidence: {str(cve.get('text', ''))[:500]}"
            )
        cve_context = "\n".join(cve_lines)

    system_prompt = """You are a senior security engineer and vulnerability analyst. Your task is to generate a comprehensive security report for a code vulnerability.

Given:
- Vulnerability type
- CVSS severity score
- Vulnerable code
- Initial vulnerability analysis

Generate a detailed JSON report with the following structure:
{
  "vuln": "vulnerability name",
  "cvss_score": 9.8,
  "executive_summary": "Detailed 2-3 sentence summary explaining what the vulnerability is, how it can be exploited, and why it's critical",
  "potential_impact": [
    "Impact item 1 (be specific to this code and vulnerability)",
    "Impact item 2",
    "Impact item 3",
    "Impact item 4 (include attack difficulty and required privileges here)"
  ],
  "attack_difficulty": "Low|Medium|High",
  "required_privileges": "None|Low|High",
  "recommended_mitigation": {
    "immediate": "Immediate action to take (1-2 sentences)",
    "short_term": "Short-term action within 1-2 weeks (1-2 sentences)",
    "long_term": "Long-term strategic action (1-2 sentences)"
  },
  "implementation_steps": [
    "Specific implementation step 1",
    "Specific implementation step 2",
    "Specific implementation step 3",
    "Specific implementation step 4"
  ],
  "patched_code": {
    "language": "python|javascript|php|java",
    "code_snippet": "FULL corrected/patched version of the original code"
  },
  "estimated_effort_hours": 6,
  "confidence": 0.85
}

Guidelines:
- Be specific to the actual code provided, not generic
- Executive summary should explain the vulnerability clearly
- Impact items should be realistic and relevant
- Mitigation should be actionable and prioritized
- Implementation steps should be concrete and technical
- Patched code must be COMPLETE and working code (not snippets)
- Estimated effort should be based on code complexity (typical range: 4-16 hours)
- Confidence based on CVSS score and analysis quality (0.7-0.95)
- Return ONLY valid JSON, no markdown, no extra text
"""

    user_prompt = f"""Vulnerability type: {vuln}
CVSS Score: {cvss_score}
Programming language: {language}

Vulnerable code:
```{language}
{code}
```

Initial analysis:
{llama_analysis}

Related CVE evidence from RAG:
{cve_context}

Generate a complete security report in JSON format."""

    try:
        resp = client.chat.completions.create(
            model=config.UPSTAGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=config.UPSTAGE_TEMPERATURE,
            max_tokens=config.REPORT_MAX_TOKENS,
            stream=False
        )
    except Exception as ex:
        raise RuntimeError(f'API call failed: {ex}')

    # Extract content
    try:
        content = resp.choices[0].message.content.strip()
    except (AttributeError, IndexError) as e:
        raise RuntimeError(f'Unexpected API response format: {e}')

    # Parse JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code block
        import re
        match = re.search(r'\{[\s\S]*\}', content)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise RuntimeError(f'Could not parse JSON from API response: {content[:200]}...')

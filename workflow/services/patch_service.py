#!/usr/bin/env python3
"""
patch_service.py

Vulnerability processing module for LlamaGuard.

Handles external LLM API calls to generate security patches for vulnerable code.
Uses configuration from config.py for all thresholds and API settings.
"""

import os
import sys
import json
from typing import Dict, Any

# Add parent directory to path for config import
service_dir = os.path.dirname(__file__)
workflow_dir = os.path.join(service_dir, '..')
sys.path.insert(0, workflow_dir)

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
# Main processing logic
# ---------------------------
def process_input(original_code: str, vuln: str, score: float, language: str) -> Dict[str, Any]:
    """
    Process vulnerability and generate patch if needed.

    Args:
        original_code: Vulnerable source code
        vuln: Vulnerability type/name
        score: Severity score (0-1)
        language: Programming language

    Returns:
        Low severity: {"vuln": ..., "decision": "ok", "message": ...}
        High severity: {"vuln": ..., "patched_code": {"language": ..., "code_snippet": ...}}
    """
    if score < config.PATCH_SCORE_THRESHOLD:
        return {
            "vuln": vuln,
            "decision": "ok",
            "message": f"Low severity (score: {score:.2f}) - monitoring recommended."
        }

    # High severity -> call external API
    resp = call_external_for_patch(vuln=vuln, code=original_code, language=language)

    # Validate response
    if not isinstance(resp, dict) or 'patched_code' not in resp:
        raise RuntimeError('Invalid API response: missing patched_code')

    patched = resp['patched_code']
    if not patched.get('code_snippet'):
        raise RuntimeError('API response contains empty code_snippet')

    return {
        "vuln": vuln,
        "patched_code": {
            "language": patched.get('language', language),
            "code_snippet": patched['code_snippet']
        }
    }

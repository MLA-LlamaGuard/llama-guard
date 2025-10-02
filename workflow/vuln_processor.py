#!/usr/bin/env python3
"""
vuln_processor.py

Simple processor that implements your requested behavior:
- Input: original_code (string), vuln (string), score (float), language (string)
- If score < 0.7 -> return a simple "ok" JSON: {vuln, decision: "ok", message}
- If score >= 0.7 -> sanitize code, call external LLM (or mock), and return {vuln, patched_code: {language, code_snippet}}

Usage:
  # mock run (no external API required)
  python vuln_processor.py --mock

  # real run with input file (input.json) and environment variables:
  export UPSTAGE_API_KEY="YOUR_KEY"
  export UPSTAGE_BASE_URL="https://api.upstage.ai/v1"   # optional
  export UPSTAGE_MODEL="solar-pro2"                  # optional
  python vuln_processor.py --input input.json

The script writes the output to vuln_report_output.json in the current folder.
"""

import os
import re
import json
import argparse
from typing import Dict, Any

# ---------------------------
# Sanitizer
# ---------------------------
def sanitize_code(code: str) -> str:
    """Very conservative sanitizer: masks strings, numbers, urls, paths, and identifiers.
    For production use, replace with language-aware parser/tokenizer.
    """
    if not isinstance(code, str):
        return ""
    # mask strings
    code = re.sub(r'(["\'])(?:(?=(\\?))\2.)*?\1', '<STR>', code)
    # mask numbers
    code = re.sub(r'\b\d+\b', '<NUM>', code)
    # mask urls
    code = re.sub(r'https?://\S+', '<URL>', code)
    # mask simple paths
    code = re.sub(r'([A-Za-z]:\\|/)[^\s\'"]+', '<PATH>', code)
    # conservative identifier masking
    tokens = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', code)
    id_map = {}
    idx = 1
    keywords = set(['if','else','for','while','return','def','class','import','from','try','except',
                    'public','private','protected','static','void','int','float','String','var','let','const','function'])
    for t in tokens:
        if t in keywords:
            continue
        if t not in id_map:
            id_map[t] = f"VAR_{idx}"
            idx += 1
    def _repl(m):
        w = m.group(0)
        return id_map.get(w, w)
    code = re.sub(r'\b[A-Za-z_][A-Za-z0-9_]*\b', _repl, code)
    return code

# ---------------------------
# External LLM call (OpenAI-compatible client)
# ---------------------------
def call_external_for_patch(vuln: str, sanitized_code: str, language: str, use_mock: bool=True) -> Dict[str, Any]:
    """If use_mock True, return a deterministic mocked patch.
    Otherwise try to call Upstage/OpenAI-compatible API using openai.OpenAI client.

    Returns a dict: {"vuln":..., "patched_code": {"language":..., "code_snippet":...}}
    """
    if use_mock:
        return {
            "vuln": vuln,
            "patched_code": {
                "language": language,
                "code_snippet": "/* patched snippet (masked) */\nFUNCTION_CALL_PARAM_BIND(VAR_1);"
            }
        }

    # Real call path: require environment variable UPSTAGE_API_KEY
    api_key = os.environ.get('UPSTAGE_API_KEY')
    if not api_key:
        raise RuntimeError('UPSTAGE_API_KEY environment variable is required for real mode')

    base_url = os.environ.get('UPSTAGE_BASE_URL', 'https://api.upstage.ai/v1')
    model = os.environ.get('UPSTAGE_MODEL', 'solar-pro2')

    # Import OpenAI SDK lazily to avoid hard dependency when using mock
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError('OpenAI SDK not installed. Install openai==1.52.2 to use real API')

    client = OpenAI(api_key=api_key, base_url=base_url)

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
        f"vulnerable_code:\n{sanitized_code}\n\n"
        "Return a single JSON object with the patched code (no extra commentary)."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system_prompt}, {"role":"user","content":user_prompt}],
            temperature=0.0,
            max_tokens=1200,
            stream=False
        )
    except Exception as ex:
        raise RuntimeError(f'API call failed: {ex}')

    # Extract content (sdk returns choices[0].message.content)
    content = None
    try:
        content = resp.choices[0].message.content
    except Exception:
        try:
            content = resp.choices[0].text
        except Exception:
            # fallback: return raw
            return {"raw_response": str(resp)}

    text = content.strip()
    # try parse JSON directly
    try:
        parsed = json.loads(text)
        return parsed
    except Exception:
        # try to find first {...} block
        m = re.search(r'(\{[\s\S]*\})', text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                raise RuntimeError('Could not parse JSON from model output')
        raise RuntimeError('Model returned non-JSON output')

# ---------------------------
# Main processing logic
# ---------------------------
def process_input(original_code: str, vuln: str, score: float, language: str, use_mock: bool=True) -> Dict[str, Any]:
    if score < 0.7:
        return {
            "vuln": vuln,
            "decision": "ok",
            "message": f"위험도 {score:.2f} 미만 — 현재 상태로 사용 가능(모니터링 권장)."
        }

    # score >= 0.7 -> call external with original code (no sanitization for better context)
    resp = call_external_for_patch(vuln=vuln, sanitized_code=original_code, language=language, use_mock=use_mock)

    # Validate minimal structure
    if not isinstance(resp, dict) or 'patched_code' not in resp:
        raise RuntimeError('External response invalid: missing patched_code')
    patched = resp['patched_code']
    if not patched.get('code_snippet'):
        raise RuntimeError('External response contains empty code_snippet')

    return {
        "vuln": vuln,
        "patched_code": {
            "language": patched.get('language', language),
            "code_snippet": patched['code_snippet']
        }
    }

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description='Simple vuln processor (sanitizer + external patcher)')
    parser.add_argument('--mock', action='store_true', help='Use mock external LLM responses')
    parser.add_argument('--input', type=str, default=None, help='Path to JSON input file')
    args = parser.parse_args()

    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            j = json.load(f)
        original_code = j.get('original_code','')
        vuln = j.get('vuln','UNKNOWN')
        score = float(j.get('score',0.0))
        language = j.get('language','')
    else:
        # sample default (demo)
        original_code = "query = \"SELECT * FROM users WHERE email='\" + email + \"'\"; db.execute(query)"
        vuln = "SQL_INJECTION"
        score = 0.85
        language = 'php'

    use_mock = args.mock or (os.environ.get('UPSTAGE_API_KEY') is None)
    if use_mock:
        print('[info] running in MOCK mode (no external API)')

    result = process_input(original_code, vuln, score, language, use_mock=use_mock)

    outpath = 'vuln_report_output.json'
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f'[success] result written to {outpath}')
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()

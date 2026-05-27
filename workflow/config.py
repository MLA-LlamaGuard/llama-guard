"""
Configuration settings for LlamaGuard vulnerability analysis workflow.

All hardcoded values and magic numbers should be defined here.
"""

import os


class Config:
    """Main configuration class for LlamaGuard workflow."""

    # ============================================================================
    # PATH SETTINGS
    # ============================================================================

    # Base directories
    WORKFLOW_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(WORKFLOW_DIR)

    # Model identifier — HuggingFace Hub ID or local directory path
    MODEL_PATH = os.environ.get('MODEL_PATH', 'cycloevan/vuln_detector')

    # CVE database paths
    CVE_DIR = os.path.join(PROJECT_DIR, "CVE")
    CVE_INDEX_PATH = os.path.join(CVE_DIR, "cve_index.faiss")
    CVE_DATA_PATH = os.path.join(CVE_DIR, "cve_data.pkl")

    # ============================================================================
    # MODEL SETTINGS
    # ============================================================================

    # LLaMA model settings
    MODEL_DTYPE = "fp16"
    MAX_NEW_TOKENS = 512
    TEMPERATURE = None  # None for greedy decoding
    DO_SAMPLE = False   # False for deterministic output
    TOP_P = None

    # ============================================================================
    # CVE & RAG SETTINGS
    # ============================================================================

    # Number of similar CVEs to retrieve
    CVE_TOP_K = 5

    # Maximum CVE text length for state storage (characters)
    CVE_TEXT_TRUNCATE_LENGTH = 500

    # ============================================================================
    # VULNERABILITY DETECTION
    # ============================================================================

    # Keywords for vulnerability detection heuristic
    VULN_KEYWORDS = [
        'vulnerability', 'vulnerable', 'injection', 'xss', 'csrf',
        'insecure', 'unsafe', 'exploit', 'cwe-', 'cve-',
        'sql injection', 'command injection', 'path traversal'
    ]

    # Keywords indicating safe code (must be specific phrases — single words like
    # "safe" or "secure" appear inside vulnerability descriptions and cause false negatives)
    SAFE_KEYWORDS = [
        'no vulnerabilities',
        'no security issues',
        'no issues detected',
        'no security concerns',
        'no vulnerabilities found',
        'code appears safe',
        'code is safe',
        'appears to be safe',
        'no vulnerability',
    ]

    # Vulnerability type detection patterns (for fallback extraction)
    VULN_PATTERNS = [
        (r'SQL\s+[Ii]njection', 'SQL Injection'),
        (r'XSS|Cross[- ]Site\s+Scripting', 'Cross-Site Scripting'),
        (r'CSRF|Cross[- ]Site\s+Request\s+Forgery', 'Cross-Site Request Forgery'),
        (r'Command\s+Injection', 'Command Injection'),
        (r'Path\s+Traversal', 'Path Traversal'),
        (r'Buffer\s+Overflow', 'Buffer Overflow'),
        (r'Code\s+Injection', 'Code Injection'),
        (r'Deserialization', 'Insecure Deserialization'),
    ]

    # ============================================================================
    # SEVERITY & SCORING
    # ============================================================================

    # CVSS severity threshold (0-10)
    # Vulnerabilities >= this score will trigger patch generation
    SEVERITY_THRESHOLD = 7

    # ============================================================================
    # REPORT GENERATION
    # ============================================================================

    # Number of related CVEs to include in LLM report context
    REPORT_MAX_RELATED_CVES = 3

    # ============================================================================
    # EXTERNAL API SETTINGS
    # ============================================================================

    # HuggingFace token (required if model is private)
    HF_TOKEN = os.environ.get('HF_TOKEN')

    # Upstage API settings (read from environment)
    UPSTAGE_API_KEY = os.environ.get('UPSTAGE_API_KEY')
    UPSTAGE_BASE_URL = os.environ.get('UPSTAGE_BASE_URL', 'https://api.upstage.ai/v1')
    UPSTAGE_MODEL = os.environ.get('UPSTAGE_MODEL', 'solar-pro2')
    UPSTAGE_TEMPERATURE = 0.0
    UPSTAGE_MAX_TOKENS = 1200       # patch-only API calls
    REPORT_MAX_TOKENS = 2000        # full security report (includes patch + analysis)

    # ============================================================================
    # LANGUAGE DETECTION
    # ============================================================================

    # Programming language detection patterns.
    # Order matters: most-specific first to avoid false matches.
    # JS generic keywords (const/let/var) also appear in other languages,
    # so use JS-specific syntax (=>, ===, require) and check Java before JS.
    LANG_PATTERNS = {
        'php': ['<?php', '<?='],
        'java': ['public class ', 'import java.', 'public static void ', 'private void '],
        'javascript': ['=>', '===', '!==', 'require(', 'module.exports', 'console.log('],
        'python': ['def ', 'import ', 'class '],  # checked last; also the default
    }

    # ============================================================================
    # WORKFLOW SETTINGS
    # ============================================================================

    # Thread ID for checkpointing
    DEFAULT_THREAD_ID = "default"

    # Output encoding
    DEFAULT_ENCODING = "utf-8"


# Singleton instance
config = Config()

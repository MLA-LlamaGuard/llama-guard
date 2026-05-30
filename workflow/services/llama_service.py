#!/usr/bin/env python3
"""
llama_service.py

LLaMA model operations and CVE search service for LlamaGuard.

Provides core functionality:
- LLaMA model loading and inference
- CVE vector database operations
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Bootstrap sys.path (services/ is one level below workflow/, so resolve manually)
_svc_dir = os.path.dirname(os.path.abspath(__file__))
_workflow_dir = os.path.dirname(_svc_dir)
_project_dir = os.path.dirname(_workflow_dir)
for _p in (_workflow_dir, _project_dir, os.path.join(_project_dir, 'llama-model')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llama_predict import resolve_dtype, build_prompt
from CVE.cve_vectordb import CVEVectorDB


def load_model(model_path: str, dtype, hf_token: str = None, subfolder: str = None):
    """
    Load fine-tuned LLaMA model for vulnerability analysis.

    Args:
        model_path: HuggingFace Hub ID (e.g. 'cycloevan/vuln_detector') or local directory
        dtype: Data type for model (torch dtype object)
        hf_token: HuggingFace token for private models (optional)
        subfolder: Optional HuggingFace repository subfolder containing model files

    Returns:
        Tuple of (tokenizer, model)

    Raises:
        RuntimeError: If model loading fails
    """
    print(f"\n[1/3] Loading LLaMA model from {model_path}...")

    try:
        load_kwargs = {"token": hf_token}
        if subfolder and not os.path.isdir(model_path):
            load_kwargs["subfolder"] = subfolder

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, **load_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto", **load_kwargs
        )
        model.eval()
        print(f"Model loaded (GPU: {torch.cuda.is_available()})")
        return tokenizer, model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")


def analyze_code(code: str, tokenizer, model, max_new_tokens: int = 512) -> str:
    """
    Analyze code for vulnerabilities using LLaMA model.

    Args:
        code: Source code to analyze
        tokenizer: HuggingFace tokenizer
        model: LLaMA model
        max_new_tokens: Maximum tokens to generate (default: 512)

    Returns:
        Vulnerability analysis text

    Raises:
        RuntimeError: If inference fails
    """
    print(f"\n[2/3] Analyzing code with LLaMA...")

    try:
        prompt = build_prompt(code)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs.input_ids.shape[1]
        generated_ids = output[0, input_len:]
        result = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print("Analysis complete")
        return result.strip()
    except Exception as e:
        raise RuntimeError(f"Code analysis failed: {e}")


def load_cve_db(index_path: str, data_path: str):
    """
    Load CVE vector database.

    Args:
        index_path: Path to FAISS index file
        data_path: Path to pickle data file

    Returns:
        CVEVectorDB instance or None if files not found

    Raises:
        RuntimeError: If database loading fails
    """
    print(f"\n[3/3] Loading CVE Vector Database...")

    if not os.path.exists(index_path):
        print(f"WARNING: CVE index not found at {index_path}")
        return None

    if not os.path.exists(data_path):
        print(f"WARNING: CVE data not found at {data_path}")
        return None

    try:
        db = CVEVectorDB()
        db.load(index_path, data_path)
        print(f"CVE DB loaded ({len(db.cve_entries)} entries)")
        return db
    except Exception as e:
        raise RuntimeError(f"Failed to load CVE database: {e}")


def search_cves(query: str, cve_db, top_k: int = 5):
    """
    Search for similar CVEs using semantic search.

    Args:
        query: Query text (vulnerability analysis)
        cve_db: CVEVectorDB instance
        top_k: Number of results to return (default: 5)

    Returns:
        List of (CVEEntry, similarity_score) tuples

    Raises:
        RuntimeError: If search fails
    """
    print(f"\n[RAG] Searching for similar CVEs (top {top_k})...")

    try:
        results = cve_db.search(query, top_k=top_k)
        print(f"Found {len(results)} similar CVEs")
        return results
    except Exception as e:
        raise RuntimeError(f"CVE search failed: {e}")

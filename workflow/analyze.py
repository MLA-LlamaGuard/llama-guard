#!/usr/bin/env python3
import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'llama-model'))

from llama_predict import resolve_dtype, build_prompt
from CVE.cve_vectordb import CVEVectorDB, CVEEntry


def parse_args():
    p = argparse.ArgumentParser(description="LlamaGuard: 코드 취약점 분석 + CVE RAG")
    p.add_argument("--model", type=str, default="../llama-model/merged-vuln-detector")
    p.add_argument("--code", type=str, help="직접 입력한 코드")
    p.add_argument("--code_file", type=str, help="분석할 코드 파일 경로")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--cve_index", type=str, default="../CVE/cve_index.faiss")
    p.add_argument("--cve_data", type=str, default="../CVE/cve_data.pkl")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--no_rag", action="store_true")
    return p.parse_args()


def load_model(model_path, dtype):
    print(f"\n[1/3] Loading LLaMA model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map="auto")
    model.eval()
    print(f"Model loaded (GPU: {torch.cuda.is_available()})")
    return tokenizer, model


def analyze_code(code, tokenizer, model, max_new_tokens):
    print(f"\n[2/3] Analyzing code with LLaMA...")
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


def load_cve_db(index_path, data_path):
    print(f"\n[3/3] Loading CVE Vector Database...")
    if not os.path.exists(index_path) or not os.path.exists(data_path):
        print("WARNING: CVE database not found")
        return None

    db = CVEVectorDB()
    db.load(index_path, data_path)
    print(f"CVE DB loaded ({len(db.cve_entries)} entries)")
    return db


def search_cves(query, cve_db, top_k):
    print(f"\n[RAG] Searching for similar CVEs (top {top_k})...")
    results = cve_db.search(query, top_k=top_k)
    print(f"Found {len(results)} similar CVEs")
    return results


def calculate_cvss_score(cve_results):
    """RAG 결과 5개의 CVSS 점수 평균 계산"""
    print(f"\n[CVSS] Calculating average CVSS score from {len(cve_results)} similar CVEs...")

    cvss_scores = []
    cve_details = []

    for cve_entry, similarity in cve_results:
        cvss_str = cve_entry.metadata.get('cvss', '')
        if cvss_str:
            try:
                # "7.2 (HIGH) [v3.1]" -> 7.2 추출
                score = float(cvss_str.split()[0])
                cvss_scores.append(score)
                cve_details.append({
                    'cve_id': cve_entry.cve_id,
                    'cvss': score,
                    'cvss_full': cvss_str,
                    'similarity': similarity
                })
            except:
                pass

    if not cvss_scores:
        return "Unable to calculate CVSS score (no valid scores found)"

    # 평균 계산
    avg_cvss = sum(cvss_scores) / len(cvss_scores)

    result = f"Average CVSS: {avg_cvss:.2f}\n\n"
    result += "Based on similar CVEs:\n"
    for detail in cve_details:
        result += f"  - {detail['cve_id']}: {detail['cvss']} - Similarity: {detail['similarity']:.3f}\n"

    print(f"Average CVSS: {avg_cvss:.2f}")
    return result


def main():
    args = parse_args()
    dtype = resolve_dtype(args.dtype)

    # 입력 코드 준비
    if args.code:
        code = args.code
        print("[Input] Direct code input")
    elif args.code_file and os.path.exists(args.code_file):
        with open(args.code_file, "r", encoding="utf-8") as f:
            code = f.read()
        print(f"[Input] Code from {args.code_file}")
    else:
        code = (
            "def login(username, password):\n"
            "    query = f\"SELECT * FROM users WHERE username='{username}' AND password='{password}'\"\n"
            "    cursor.execute(query)\n"
            "    return cursor.fetchone()\n"
        )
        print("[Input] Using default example (SQL Injection)")

    # LLaMA 모델 로드 및 분석
    tokenizer, model = load_model(args.model, dtype)
    llama_result = analyze_code(code, tokenizer, model, args.max_new_tokens)

    # RAG 쿼리 (취약점 있다고 가정)
    cve_results = None
    cvss_result = None
    if not args.no_rag:
        cve_db = load_cve_db(args.cve_index, args.cve_data)
        if cve_db is not None:
            cve_results = search_cves(llama_result, cve_db, args.top_k)

            # CVSS 점수 계산 (RAG 결과 평균)
            if cve_results:
                cvss_result = calculate_cvss_score(cve_results)

    # 결과 출력
    print("\n" + "="*80)
    print("LLAMA ANALYSIS:")
    print("-" * 80)
    print(llama_result)
    print("="*80)

    if cvss_result:
        print("\nCVSS SCORE ANALYSIS:")
        print("-" * 80)
        print(cvss_result)
        print("="*80)

    if cve_results:
        print("\nRELATED CVEs:")
        print("-" * 80)
        for idx, (cve_entry, score) in enumerate(cve_results, 1):
            print(f"\n[{idx}] {cve_entry.cve_id} (Similarity: {score:.4f})")

            # Metadata 출력
            if 'cvss' in cve_entry.metadata:
                print(f"    CVSS: {cve_entry.metadata['cvss']}")
            if 'cwe' in cve_entry.metadata:
                print(f"    CWE: {cve_entry.metadata['cwe']}")
            if 'published' in cve_entry.metadata:
                print(f"    Published: {cve_entry.metadata['published']}")

            # 전체 텍스트 출력
            print(f"\n    Full CVE Text:")
            print("    " + "-" * 76)
            for line in cve_entry.text.split('\n'):
                print(f"    {line}")
            print("    " + "-" * 76)
        print("="*80)


if __name__ == "__main__":
    main()

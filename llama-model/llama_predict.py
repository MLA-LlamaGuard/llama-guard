import os
import re
import json
import time
import argparse
from tqdm import tqdm
import random
import numpy as np
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# (선택) CPU/GPU 사용량 요약용
import platform
use_resource = False
try:
    import resource  # Unix
    use_resource = True
except Exception:
    pass

use_psutil = False
try:
    import psutil
    use_psutil = True
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser(description="KorQuAD v1.0 dev prediction (Llama Instruct)")
    p.add_argument("--dev_json", type=str, required=True, help="KorQuAD_v1.0_dev.json 경로")
    p.add_argument("--model", type=str, required=True, help="모델 디렉토리/이름 (베이스 또는 병합 후)")
    p.add_argument("--out", type=str, required=True, help="저장할 예측 JSON 경로 (qid -> answer)")
    p.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", 16)))
    p.add_argument("--max_input_tokens", type=int, default=int(os.getenv("MAX_INPUT_TOKENS", 1536)))
    p.add_argument("--max_new_tokens", type=int, default=int(os.getenv("MAX_NEW_TOKENS", 32)))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attn_impl", type=str, default="sdpa", choices=["auto", "flash_attention_2", "sdpa", "eager"])
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "auto"])
    return p.parse_args()


def set_seed_everywhere(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dev_korquad(dev_json_path: str):
    """
    KorQuAD v1.0 dev 파일을 robust 하게 로드:
    - 단일 JSON 객체: {"version": "...", "data":[...]}
    - JSONL(줄단위) 또는 여러 JSON 객체가 이어진 파일: 각 객체에서 "data"를 모아 결합
    반환값: list of articles (= dataset["data"])
    """
    with open(dev_json_path, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    # 1) 일반적인 단일 JSON 케이스
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict) and "data" in obj:
            return obj["data"]
        # 혹시 최상위가 리스트인 변형 포맷이면 그대로 반환 시도 (기사 리스트로 가정)
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass  # 아래에서 라인단위 파싱 시도

    # 2) JSONL/여러 JSON 객체가 줄마다 있는 케이스
    data = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            # 공백/잘린 라인 무시
            continue
        if isinstance(o, dict) and "data" in o:
            # 정규 KorQuAD 조각이라면 data 배열을 이어붙임
            data.extend(o["data"])
        else:
            # 혹시 이미 article 구조(dict)인 경우 그대로 추가 (호환성 모드)
            data.append(o)
    if not data:
        raise ValueError(f"[load_dev_korquad] 지원하지 않는 포맷이거나 파싱 실패: {dev_json_path}")
    return data

def iter_korquad_examples(dev_list):
    """
    dev_list가
      A) 원본 KorQuAD(JSON: data->paragraphs->qas) 이면: article/paragraph/qas 계층을 따라가고
      B) 평탄화(context/question/answers/id) 형식이면: 각 레코드를 그대로 사용
    (qid, context, question) 을 yield 합니다.
    """
    # 케이스 A: 원본 구조 감지
    if len(dev_list) > 0 and isinstance(dev_list[0], dict) and "paragraphs" in dev_list[0]:
        for article in dev_list:
            for para in article["paragraphs"]:
                ctx = para["context"]
                for qa in para["qas"]:
                    yield qa["id"], ctx, qa["question"]
        return

    # 케이스 B: 평탄화 구조(validation 등)
    for ex in dev_list:
        # ex: {"id":..., "context":..., "question":..., "answers": {...}}
        if all(k in ex for k in ("id", "context", "question")):
            yield ex["id"], ex["context"], ex["question"]


def build_prompt(tok: AutoTokenizer, context: str, question: str) -> str:
    # 학습 때와 동일한 chat template 사용 (system+user, add_generation_prompt=True)
    messages = [
        {"role": "system", "content": "다음 지문을 보고 질문의 정답만 한글로 출력하세요."},
        {"role": "user",   "content": f"지문: {context}\n질문: {question}"},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def resolve_dtype(dtype_flag: str) -> torch.dtype:
    if dtype_flag == "fp32":
        return torch.float32
    if dtype_flag == "bf16":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    # auto
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def load_model(model_path: str, attn_impl: str, device_str: str, dtype_flag: str):
    dtype = resolve_dtype(dtype_flag)

    # auto는 재현성 우선 관점에서 sdpa로 맵핑
    if attn_impl == "auto":
        attn_impl = "sdpa"

    # device_map = "cuda" if (device_str == "cuda" and torch.cuda.is_available()) else "cpu"

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=attn_impl,
    )


def main():
    args = parse_args()
    set_seed_everywhere(args.seed)

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    print("== SETTINGS ==")
    print("DEV JSON :", args.dev_json)
    print("MODEL    :", args.model)
    print("OUT PATH :", args.out)
    print("BATCH    :", args.batch_size)
    print("MAX_IN   :", args.max_input_tokens)
    print("MAX_NEW  :", args.max_new_tokens)
    print("SEED     :", args.seed)
    print()

    print("loading model/tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        
    tok.padding_side = "left"
    tok.truncation_side = "left"

    model = load_model(args.model, args.attn_impl, args.device, args.dtype)
    model.eval()

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,  # greedy
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )

    print("reading dev file:", args.dev_json)
    dev = load_dev_korquad(args.dev_json)

    preds = {}
    wall_start = time.perf_counter()
    cuda_time_ms_total = 0.0
    num_batches = 0
    num_examples = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def batch_generate(prompts):
        nonlocal cuda_time_ms_total, num_batches
        # 토크나이즈
        tok_out = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(model.device)

        # CUDA 타이머
        if torch.cuda.is_available():
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_evt.record()

        with torch.inference_mode():
            out = model.generate(**tok_out, generation_config=gen_cfg)

        if torch.cuda.is_available():
            end_evt.record()
            torch.cuda.synchronize()
            batch_ms = start_evt.elapsed_time(end_evt)
            cuda_time_ms_total += float(batch_ms)

        num_batches += 1

        # 입력 길이별로 생성된 부분만 디코딩
        input_lens = (tok_out["input_ids"] != tok.pad_token_id).sum(dim=1)
        gens = []
        for i in range(out.size(0)):
            gen_ids = out[i, input_lens[i]:]
            text = tok.decode(gen_ids, skip_special_tokens=True)
            # 첫 줄만 추출 (공식 스크립트가 정규화 처리)
            ans = re.split(r"[\r\n]", text.strip())[0]
            ans = ans.strip().strip("'\"()[]「」")
            gens.append(ans)
        return gens

    qids, prompts = [], []
    for qid, context, question in tqdm(iter_korquad_examples(dev)):
        qids.append(qid)
        prompts.append(build_prompt(tok, context, question))

        if len(prompts) == args.batch_size:
            for _qid, ans in zip(qids, batch_generate(prompts)):
                preds[_qid] = ans
            num_examples += len(prompts)
            qids, prompts = [], []
            
    if prompts:
        for _qid, ans in zip(qids, batch_generate(prompts)):
            preds[_qid] = ans
        num_examples += len(prompts)

    # 저장
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False)

    # 요약 출력
    wall_end = time.perf_counter()
    wall_secs = wall_end - wall_start

    print("saved:", args.out)
    print("\n===== PROFILING SUMMARY =====")
    print(f"Examples processed     : {num_examples}")
    print(f"Batches processed      : {num_batches}")
    print(f"Wall time (s)          : {wall_secs:.3f}")
    if torch.cuda.is_available():
        peak_res = torch.cuda.max_memory_reserved()
        print(f"CUDA time total (ms)   : {cuda_time_ms_total:.3f}")
        print(f"GPU peak reserved (MB) : {peak_res/1024/1024:.2f}")
    else:
        print("CUDA not available -> CUDA time/GPU memory not measured.")
    if use_resource:
        # Linux/Unix: KB 단위 (일부 OS는 단위 다를 수 있음)
        print(f"CPU peak RSS          : {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} KB")
    elif use_psutil:
        print(f"CPU current RSS       : {int(psutil.Process().memory_info().rss/1024)} KB (psutil)")
    print("=============================\n")


if __name__ == "__main__":
    main()

import os
import time
import csv
import psutil
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from peft import PeftModel
import random
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

# =========================
# 콜백들: 런타임 로그 + 시간 예산 컷
# =========================
class MemoryTimingCallback(TrainerCallback):
    def __init__(self, output_dir, log_every=50):
        self.output_dir = output_dir
        self.log_every = log_every
        self.proc = psutil.Process(os.getpid())
        self.csv_path = os.path.join(output_dir, "train_runtime_metrics.csv")
        self.wall_start = None
        self.cpu_peak_rss = 0
        self.max_cuda_ms = 0.0
        self.max_gpu_mem = 0
        self.step_start = None
        self.cuda_start = None
        self.gpu_available = torch.cuda.is_available()

        if self.gpu_available:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        os.makedirs(output_dir, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "global_step",
                "wall_step_sec",
                "cuda_step_ms",
                "gpu_mem_allocated_mb",
                "gpu_mem_reserved_mb",
                "cpu_rss_mb",
            ])

    def on_train_begin(self, args, state, control, **kwargs):
        self.wall_start = time.perf_counter()

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start = time.perf_counter()
        if self.gpu_available:
            self.cuda_start = torch.cuda.Event(enable_timing=True)
            self.cuda_end = torch.cuda.Event(enable_timing=True)
            self.cuda_start.record()

    def on_step_end(self, args, state, control, **kwargs):
        wall_step_sec = time.perf_counter() - self.step_start if self.step_start else 0.0

        cuda_step_ms = 0.0
        if self.gpu_available:
            self.cuda_end.record()
            torch.cuda.synchronize()
            cuda_step_ms = self.cuda_start.elapsed_time(self.cuda_end)

        if self.gpu_available:
            gpu_alloc = torch.cuda.max_memory_allocated() / (1024**2)
            gpu_resvd = torch.cuda.max_memory_reserved() / (1024**2)
            torch.cuda.reset_peak_memory_stats()
        else:
            gpu_alloc = gpu_resvd = 0.0

        cpu_rss_mb = self.proc.memory_info().rss / (1024**2)
        self.cpu_peak_rss = max(self.cpu_peak_rss, cpu_rss_mb)
        self.max_cuda_ms = max(self.max_cuda_ms, cuda_step_ms)
        self.max_gpu_mem = max(self.max_gpu_mem, gpu_resvd)

        if state.global_step % self.log_every == 0 or state.global_step == state.max_steps:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    state.global_step,
                    round(wall_step_sec, 6),
                    round(cuda_step_ms, 3),
                    round(gpu_alloc, 2),
                    round(gpu_resvd, 2),
                    round(cpu_rss_mb, 2),
                ])

    def on_train_end(self, args, state, control, **kwargs):
        total_wall = time.perf_counter() - self.wall_start if self.wall_start else 0.0
        print("\n=== [RUNTIME SUMMARY] ===")
        print(f"Total wall time (sec): {total_wall:.2f}")
        print(f"Peak CPU RSS (MB):     {self.cpu_peak_rss:.2f}")
        print(f"Peak GPU reserved (MB):{self.max_gpu_mem:.2f}")
        print(f"Max per-step CUDA (ms):{self.max_cuda_ms:.2f}")
        print(f"CSV saved to: {self.csv_path}")

class TimeBudgetCallback(TrainerCallback):
    """학습 시작 후 max_minutes가 지나면 훈련을 종료"""
    def __init__(self, max_minutes=25):
        self.max_sec = max_minutes * 60
        self.t0 = None
    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.perf_counter()
    def on_step_end(self, args, state, control, **kwargs):
        if time.perf_counter() - self.t0 > self.max_sec:
            control.should_training_stop = True
            print(f"\n[TimeBudgetCallback] Reached {self.max_sec/60:.1f} min. Stopping training.")

# =========================
# 경로/모델 설정
# =========================
model_name = "./llama-3.2-1B-Instruct"
output_dir = "./llama-3.2-1B-Instruct-lora"
lora_adapter_dir = f"{output_dir}/lora-adapter"
os.makedirs(output_dir, exist_ok=True)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

# =========================
# KorQuAD 로드 & 평탄화
# =========================
print("1) KorQuAD_v1.0 로드...")
data_dir = "./KorQuAD_v1.0"
train_path = f"{data_dir}/korquad_train.json"
val_path = f"{data_dir}/korquad_validation.json"

raw = load_dataset("json", data_files={"train": train_path, "validation": val_path,})

train_base = raw["train"]
valid_base = raw["validation"]

# =========================
# 모델/토크나이저 로드 (4bit QLoRA)
# =========================
print("\n2) 모델/토크나이저 로드 (4bit NF4 + bf16 계산)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # 모델을 4비트 양자화로 로드합니다.
    bnb_4bit_quant_type="nf4",             # 양자화 유형을 "nf4"로 설정합니다. "nf4"는 Normal Float 4를 의미하며, 4비트 양자화에서 높은 정밀도를 제공합니다.
    bnb_4bit_compute_dtype=torch.bfloat16, # 계산 시 사용할 데이터 타입을 bfloat16으로 설정합니다. 이는 메모리 효율성과 계산 속도 간의 균형을 제공합니다.
    bnb_4bit_use_double_quant=False,       # 이중 양자화를 사용하지 않도록 설정합니다. 이중 양자화는 추가적인 정밀도를 제공하지만, 여기서는 비활성화되어 있습니다.
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

try:
    model.config.attn_implementation = "sdpa"
except Exception:
    pass

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# =========================
# 프롬프트 생성 (모델의 채팅 템플릿 사용)
# =========================
print("\n3) 프롬프트 생성(format_prompt)...")

def formatting_func(examples):
    texts = []
    for ctx, q, ans in zip(examples["context"], examples["question"], examples["answers"]):
        messages = [
            {"role": "system", "content": "다음 지문을 보고 질문의 정답만 한글로 출력하세요."},
            {"role": "user",   "content": f"지문: {ctx}\n질문: {q}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = ans["text"][0].strip() + tokenizer.eos_token
        texts.append(prompt + response)  # 단일 문자열로 반환해도 됨
    return {"text": texts}

# TRL이 기대하는 키로 매핑
def to_trl_schema(dataset):
    return dataset.map(
        formatting_func,
        batched=True,
        remove_columns=dataset.column_names,
    )

# =========================
# LoRA 설정
# =========================
print("\n4) LoRA 설정...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
)

# =========================
# 데이터 서브샘플 & Packing 준비
# =========================
print("\n5) 데이터 서브샘플링 & Packing 준비...")
MAX_TRAIN_SAMPLES = 8000
train_base = train_base.shuffle(seed=42).select(
    range(min(MAX_TRAIN_SAMPLES, len(train_base)))
)
eval_ds = None
MAX_SEQ_LENGTH = 1024
if hasattr(tokenizer, "model_max_length"):
    MAX_SEQ_LENGTH = min(MAX_SEQ_LENGTH, tokenizer.model_max_length)


# =========================
# 트레이닝 아규먼트 (SFTConfig 사용)
# =========================
print("\n6) 트레이닝 아규먼트 설정...")
training_arguments = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="paged_adamw_8bit",
    fp16=False,
    bf16=True,
    max_steps=240,
    num_train_epochs=1,
    logging_steps=50,
    save_strategy="no",
    eval_strategy="no",
    report_to="none",
    lr_scheduler_type="cosine",
    learning_rate=2e-4,
    warmup_ratio=0.03,
    dataloader_num_workers=2,
    remove_unused_columns=False,
    packing=True,
    max_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    seed=SEED,
    data_seed=SEED,
)

# =========================
# SFTTrainer 초기화 & 학습
# =========================
print("\n7) SFTTrainer 초기화...")
metrics_cb = MemoryTimingCallback(output_dir=output_dir, log_every=50)
time_cb = TimeBudgetCallback(max_minutes=25)

train_proc = train_base.map(
    formatting_func,
    batched=True,
    remove_columns=train_base.column_names,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_proc,
    eval_dataset=eval_ds,
    peft_config=peft_config,
    args=training_arguments,
    callbacks=[metrics_cb, time_cb],
)

print("\n8) 파인튜닝 시작...")
trainer.train()

# =========================
# LoRA 어댑터/토크나이저 저장
# =========================
print("\n9) LoRA 어댑터 저장...")
os.makedirs(lora_adapter_dir, exist_ok=True)
trainer.save_model(lora_adapter_dir)
tokenizer.save_pretrained(output_dir)
print(f"LoRA 어댑터가 '{lora_adapter_dir}'에 저장되었습니다.")
print(f"토크나이저가 '{output_dir}'에 저장되었습니다.")






base_dir = "./llama-3.2-1B-Instruct"                     # 베이스(학습 전) 디렉토리
lora_dir = "./llama-3.2-1B-Instruct-lora/lora-adapter"   # 방금 저장한 LoRA 어댑터
out_dir  = "./merged-1B-instruct-ko"                     # 병합 결과 저장 디렉토리

# CPU에서 병합 권장(메모리 여유). GPU여도 무방
dtype = torch.bfloat16  # 또는 torch.float16
base = AutoModelForCausalLM.from_pretrained(
    base_dir, torch_dtype=dtype, device_map="cpu", trust_remote_code=True
)
tok = AutoTokenizer.from_pretrained(base_dir, use_fast=True, trust_remote_code=True)

# LoRA 로드 후 병합
model = PeftModel.from_pretrained(base, lora_dir)
merged = model.merge_and_unload()   # ← 핵심: 가중치에 어댑터를 굽습니다.

# 저장 (safetensors)
merged.save_pretrained(out_dir, safe_serialization=True)
tok.save_pretrained(out_dir)
print(f"Merged model saved to: {out_dir}")

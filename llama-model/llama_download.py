from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os


def download_and_save(model_id: str, save_dir: str, hf_token: str = None):
    has_cuda = torch.cuda.is_available()
    dtype = torch.float16 if has_cuda else torch.float32
    device_map = "auto" if has_cuda else None

    print(f"\n[다운로드 시작] {model_id}")
    print(f" - GPU 사용: {has_cuda}")
    print(f" - 저장 경로: {save_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=hf_token)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        token=hf_token,
    )

    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)

    print(f"[완료] 모델과 토크나이저가 '{save_dir}'에 저장되었습니다.")


if __name__ == "__main__":
    token = os.environ.get("HF_TOKEN")
    # __file__ 기준 경로 사용 — CWD에 무관하게 항상 프로젝트 루트의 models/ 에 저장
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _save_dir = os.path.normpath(os.path.join(_script_dir, "..", "models", "vuln_detector"))
    download_and_save(
        model_id="cycloevan/vuln_detector",
        save_dir=_save_dir,
        hf_token=token,
    )


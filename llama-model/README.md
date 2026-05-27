# LLaMA Model — 파인튜닝 및 추론

`meta-llama/Llama-3.2-1B-Instruct` 기반 코드 취약점 분석 모델 QLoRA 파인튜닝 가이드.

---

## 사전 요구사항

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) — 프로젝트 루트에서 관리
- NVIDIA GPU (CUDA 11.8+, QLoRA / bitsandbytes 필수)

---

## 환경 설정

프로젝트 루트에서 실행:

```bash
# 학습 의존성 포함 설치
uv sync --extra train
```

---

## 실행 순서

### Step 1. Base 모델 다운로드

```bash
HF_TOKEN=your_token python llama-model/llama_download.py
```

저장 위치: `models/vuln_detector/`

학습용 base 모델 (`meta-llama/Llama-3.2-1B-Instruct`)은 별도로 다운로드:

```bash
# llama-model/ 디렉터리 안에 저장
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir llama-model/llama-3.2-1B-Instruct
```

### Step 2. QLoRA 파인튜닝

```bash
uv run python llama-model/llama_fine_tuning.py
```

- **입력 모델**: `llama-model/llama-3.2-1B-Instruct/`
- **학습 데이터**: `llama-model/data/secure_programming_dpo_flat.json`
- **출력**: `llama-model/merged-vuln-detector/` (LoRA 병합 완료)

### Step 3. 독립 추론 CLI

```bash
# Hub 모델 사용 (기본)
uv run python llama-model/llama_predict.py \
  --model cycloevan/vuln_detector \
  --code "def login(u, p): query = f'SELECT * FROM users WHERE u={u}'"

# 로컬 모델 사용
uv run python llama-model/llama_predict.py \
  --model llama-model/merged-vuln-detector \
  --code_file path/to/code.py
```

> 전체 파이프라인(RAG + 보고서 생성)은 `workflow/graph.py` 사용 권장.

---

## 모델 정보

| 항목 | 값 |
|------|----|
| Base 모델 | `meta-llama/Llama-3.2-1B-Instruct` |
| Hub 모델 | [`cycloevan/vuln_detector`](https://huggingface.co/cycloevan/vuln_detector) |
| 학습 방식 | QLoRA (4-bit NF4, r=16, α=32) |
| 타깃 모듈 | q_proj, k_proj, v_proj, o_proj |
| 최대 스텝 | 240 (TimeBudgetCallback 30분 제한) |
| 라이선스 | Apache 2.0 |

---

## 성능 평가

| 모델 | ROUGE-L F1 | BLEU |
|------|-----------|------|
| Llama-3.2-1B-Instruct (베이스) | 0.0933 | 0.0061 |
| merged-vuln-detector (파인튜닝) | 0.1335 | 0.0219 |

평가 데이터셋: [`doss1232/vulnerable-code`](https://huggingface.co/datasets/doss1232/vulnerable-code)

---

## 학습 데이터셋

| 파일 | 용도 |
|------|------|
| `data/secure_programming_dpo_flat.json` | 학습용 (최대 5,000샘플) |
| `data/mydata_train.json` | 추가 학습 데이터 |

원본: [`CyberNative/Code_Vulnerability_Security_DPO`](https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO)

> `llama-model/data/` 디렉터리는 `.gitignore` 처리 (`data/` 규칙 적용).

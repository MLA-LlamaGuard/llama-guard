# LlamaGuard

파인튜닝된 LLaMA 모델과 CVE 기반 RAG를 활용한 코드 보안 취약점 분석 시스템.

---

## 개요

소스 코드를 입력하면 2단계 파이프라인으로 보안 취약점을 분석하고 패치를 제안합니다.

1. **탐지** — 파인튜닝된 LLaMA 3.2-1B 모델(`cycloevan/vuln_detector`)이 코드를 스캔하여 잠재적 취약점을 식별합니다.
2. **보강 및 보고** — 탐지된 취약점을 FAISS 기반 CVE 데이터베이스(RAG)와 매칭합니다. CVSS ≥ 7인 고위험 취약점은 Upstage Solar Pro 2가 상세 보안 보고서와 패치를 생성합니다.

---

## 아키텍처

```
입력 코드
    │
    ▼
┌─────────────────────────┐
│  initial_analysis_node  │  ← cycloevan/vuln_detector (LLaMA 3.2-1B)
└─────────────────────────┘
    │
    ▼
detection_branch
    │
    ├─ 미탐지 ────────────────────────────────────────┐
    │                                                  │
    └─ 탐지됨                                          │
         │                                             │
         ▼                                             │
    ┌──────────┐                                       │
    │ rag_node │  ← FAISS CVE 벡터 검색                │
    └──────────┘                                       │
         │                                             │
         ▼                                             │
    ┌──────────────────────┐                           │
    │ cvss_calculation_node│  ← CVE 평균 CVSS 산출     │
    └──────────────────────┘                           │
         │                                             │
         ▼                                             ▼
    ┌──────────────────────────────────────────────────────┐
    │              report_generation_node                  │
    │  CVSS < 7  → 기본 마크다운 보고서                    │
    │  CVSS ≥ 7  → Solar Pro 2 상세 보고서 + 패치 제안     │
    └──────────────────────────────────────────────────────┘
         │
         ▼
      최종 보고서 출력
```

---

## 요구사항

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) (패키지 매니저)
- [Upstage API 키](https://console.upstage.ai) (고위험 보고서 생성 시 필요)

---

## 설치 및 환경 설정

### 1. uv 설치

```bash
# macOS / Linux
brew install uv
# 또는
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 의존성 설치

```bash
# 추론 + 워크플로우만 (기본)
uv sync

# 학습 의존성 포함 (QLoRA 파인튜닝 시)
uv sync --extra train
```

### 3. 환경변수 설정

```bash
cp .env.example .env
```

`.env` 파일에 키를 입력합니다:

```env
# 고위험 보고서 생성에 필요
UPSTAGE_API_KEY=your_upstage_key_here

# cycloevan/vuln_detector가 private인 경우에만 필요
HF_TOKEN=your_hf_token_here
```

### 4. CVE 벡터 데이터베이스 구축 (최초 1회)

```bash
# Step 1 — NVD에서 CVE 데이터 다운로드 (수 분 소요)
python CVE/cve_downloader.py --output CVE/cve_database.txt

# Step 2 — FAISS 인덱스 빌드
python CVE/cve_vectordb.py \
    --input CVE/cve_database.txt \
    --index-output CVE/cve_index.faiss \
    --data-output CVE/cve_data.pkl
```

---

## 실행 방법

### 코드 직접 입력

```bash
uv run python workflow/graph.py --code "
def login(username, password):
    query = f\"SELECT * FROM users WHERE username='{username}' AND password='{password}'\"
    cursor.execute(query)
    return cursor.fetchone()
"
```

### 파일 분석

```bash
uv run python workflow/graph.py --code_file path/to/your/code.py
```

### 보고서 파일로 저장

```bash
uv run python workflow/graph.py --code_file path/to/your/code.py --output report.md
```

---

## 보고서 형식

| CVSS 범위 | 보고서 내용 |
|-----------|-------------|
| 취약점 미탐지 | SAFE 상태 + LLaMA 분석 결과 |
| 0 – 6 (낮음 / 보통) | 취약점 목록 + 초기 분석 + 모니터링 권고 |
| 7 – 10 (높음 / 심각) | 요약, 영향 분석, 완화 방안, 패치 제안 (Solar Pro 2) |

---

## 사용 모델

### 탐지 모델 (취약점 분석)

**[`cycloevan/vuln_detector`](https://huggingface.co/cycloevan/vuln_detector)**

`meta-llama/Llama-3.2-1B-Instruct` 기반으로 QLoRA 파인튜닝한 코드 취약점 탐지 모델.

| 항목 | 값 |
|------|----|
| 베이스 모델 | [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |
| 학습 방식 | QLoRA (4-bit NF4, LoRA r=16, α=32, dropout=0.1) |
| 타깃 모듈 | q_proj, k_proj, v_proj, o_proj |
| 라이선스 | Apache 2.0 |
| ROUGE-L F1 | 0.1335 (베이스 0.0933 대비 향상) |
| BLEU | 0.0219 (베이스 0.0061 대비 향상) |
| 평가 데이터셋 | [`doss1232/vulnerable-code`](https://huggingface.co/datasets/doss1232/vulnerable-code) |

Hub에서 기본 로드되며, 로컬 다운로드도 가능합니다:

```bash
python llama-model/llama_download.py
```

다운로드 위치: `models/vuln_detector/`. 로컬 모델 사용 시 `.env`에 `MODEL_PATH=models/vuln_detector` 설정.

### CVE 임베딩 모델

**[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)**

FAISS CVE 벡터 DB 구축 및 유사 CVE 검색에 사용.

| 항목 | 값 |
|------|----|
| 임베딩 차원 | 384 |
| 유사도 | 코사인 유사도 (FAISS IndexFlatIP) |
| 토큰 한도 | 256 tokens (CVE 설명 핵심부만 임베딩) |

### 보고서 생성 모델 (고위험 취약점)

**Upstage Solar Pro 2** — CVSS ≥ 7인 취약점에 대해 상세 보안 보고서와 패치를 생성.

- API: [console.upstage.ai](https://console.upstage.ai) (OpenAI 호환)
- 요금: 별도 API 키 필요 (`UPSTAGE_API_KEY`)

---

## 사용 데이터셋

### 학습 데이터

| 데이터셋 | HuggingFace 링크 | 용도 |
|---------|-----------------|------|
| Code_Vulnerability_Security_DPO | [`CyberNative/Code_Vulnerability_Security_DPO`](https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO) | QLoRA 파인튜닝 (최대 5,000샘플) |
| mydata_train.json | 커스텀 (비공개) | 추가 학습 데이터 |

### 평가 데이터

| 데이터셋 | HuggingFace 링크 | 용도 |
|---------|-----------------|------|
| vulnerable-code | [`doss1232/vulnerable-code`](https://huggingface.co/datasets/doss1232/vulnerable-code) | ROUGE-L / BLEU 성능 평가 |

### CVE 데이터

NVD (National Vulnerability Database) API에서 실시간 수집:

```bash
python CVE/cve_downloader.py --output CVE/cve_database.txt
```

- 출처: [nvd.nist.gov](https://nvd.nist.gov/developers/vulnerabilities)
- 형식: CVE ID, CVSS 점수, CWE, 설명, 패치 코드(선택)

---

## 프로젝트 구조

```
llama-guard/
├── CVE/
│   ├── cve_downloader.py           # NVD API에서 CVE 다운로드
│   ├── cve_vectordb.py             # FAISS 벡터 DB 빌드/검색
│   ├── cve_query_test.py           # CVE 검색 테스트 (CVE/ 디렉터리에서 실행)
│   └── filter_with_patches.py      # 패치 있는 CVE 필터링
├── llama-model/
│   ├── data/                       # 학습 데이터 (gitignore)
│   │   ├── secure_programming_dpo_flat.json
│   │   └── mydata_train.json
│   ├── llama_fine_tuning.py        # QLoRA SFT 학습 스크립트
│   ├── llama_predict.py            # 독립 실행 추론 CLI
│   └── llama_download.py           # HuggingFace Hub 모델 다운로드
├── models/                         # 로컬 모델 저장소 (gitignore)
│   └── vuln_detector/              # llama_download.py 실행 결과
├── workflow/
│   ├── config.py                   # 설정값 중앙 관리
│   ├── graph.py                    # LangGraph 워크플로우 + CLI 진입점
│   ├── nodes.py                    # 각 노드 구현
│   ├── path_setup.py               # sys.path 부트스트랩
│   ├── state.py                    # AgentState 정의
│   └── services/
│       ├── llama_service.py        # LLaMA 로딩/추론
│       └── patch_service.py        # Upstage API 호출
├── .env.example                    # 환경변수 템플릿
├── pyproject.toml                  # uv 의존성 관리
└── uv.lock
```

---

## 학습 (선택사항)

파인튜닝을 재현하거나 계속하려면:

```bash
# 학습 의존성 설치
uv sync --extra train

# 베이스 모델 다운로드 (llama-model/ 에 저장)
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir llama-model/llama-3.2-1B-Instruct

# QLoRA 파인튜닝 실행
uv run python llama-model/llama_fine_tuning.py
```

학습 설정 (`llama_fine_tuning.py`):
- 베이스 모델: `meta-llama/Llama-3.2-1B-Instruct`
- 양자화: 4-bit NF4 (QLoRA)
- LoRA: r=16, α=32, 타깃 모듈: q/k/v/o_proj
- 데이터셋: `llama-model/data/secure_programming_dpo_flat.json`
- 출력: `llama-model/merged-vuln-detector/`

---

## 참고 자료 

- [NVD CVE API](https://nvd.nist.gov/developers/vulnerabilities)
- [Upstage Solar Pro 2](https://console.upstage.ai)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [FAISS](https://github.com/facebookresearch/faiss)

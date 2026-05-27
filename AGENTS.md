# AGENTS.md — LlamaGuard 프로젝트 가이드

AI 에이전트 및 개발자를 위한 코드베이스 참조 문서입니다.

---

## 프로젝트 개요

소스 코드를 입력받아 보안 취약점을 분석하고 패치를 제안하는 2단계 파이프라인.

| 단계 | 역할 | 사용 모델 |
|------|------|-----------|
| 탐지 | 코드에서 취약점 여부 판정 | `cycloevan/vuln_detector` (LLaMA 3.2-1B, 파인튜닝) |
| 보강 | 관련 CVE 검색 + CVSS 산출 | FAISS 벡터 DB (RAG) |
| 보고서 | CVSS < 7: 기본 마크다운 / CVSS ≥ 7: 상세 보고서 + 패치 | Upstage Solar Pro 2 |

---

## 사용 모델

### 1. 탐지 모델 — [`cycloevan/vuln_detector`](https://huggingface.co/cycloevan/vuln_detector)

| 항목 | 값 |
|------|----|
| 베이스 | [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |
| 학습 방식 | QLoRA (4-bit NF4, LoRA r=16, α=32, dropout=0.1) |
| 타깃 모듈 | q_proj, k_proj, v_proj, o_proj |
| 라이선스 | Apache 2.0 |

**성능 평가** (평가셋: [`doss1232/vulnerable-code`](https://huggingface.co/datasets/doss1232/vulnerable-code))

| 모델 | ROUGE-L F1 | BLEU |
|------|-----------|------|
| Llama-3.2-1B-Instruct (베이스) | 0.0933 | 0.0061 |
| merged-vuln-detector (파인튜닝) | 0.1335 | 0.0219 |

**모델 로딩 방식**

```
기본 (Hub 자동 다운로드)
  MODEL_PATH=cycloevan/vuln_detector   ← .env 미설정 시 기본값

로컬 사용 (오프라인 / 속도 최적화)
  1. python llama-model/llama_download.py 실행
  2. models/vuln_detector/ 에 저장됨
  3. .env에 MODEL_PATH=models/vuln_detector 설정
```

`models/` 디렉터리는 `.gitignore` 처리 — 모델 가중치는 커밋되지 않음.

### 2. CVE 임베딩 모델 — [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

FAISS 벡터 DB 구축 및 유사 CVE 검색에 사용 (`CVE/cve_vectordb.py`).

| 항목 | 값 |
|------|----|
| 임베딩 차원 | 384 |
| 유사도 | 코사인 유사도 (IndexFlatIP + L2 정규화) |
| 토큰 한도 | 256 tokens — CVE 설명 핵심부만 임베딩 (`_embedding_text()`) |
| 자동 다운로드 | `sentence-transformers` 라이브러리가 최초 실행 시 캐시 |

> `build_index()` 호출 시 `--- Code from` 행 이후 스크래핑 코드 제거 후 임베딩.
> `entry.text` 전체는 보고서 출력에 보존됨.

### 3. 보고서 생성 모델 — Upstage Solar Pro 2

CVSS ≥ 7인 고위험 취약점에 대해 상세 보고서 + 패치 생성 (`services/patch_service.py`).

| 항목 | 값 |
|------|----|
| API | OpenAI 호환 (`https://api.upstage.ai/v1`) |
| 인증 | `UPSTAGE_API_KEY` (`.env` 설정) |
| 참고 | [console.upstage.ai](https://console.upstage.ai) |

---

## 사용 데이터셋

### 학습 데이터

| 파일 | HuggingFace 원본 | 용도 |
|------|-----------------|------|
| `data/secure_programming_dpo_flat.json` | [`CyberNative/Code_Vulnerability_Security_DPO`](https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO) | QLoRA SFT 학습 (최대 5,000샘플) |
| `data/mydata_train.json` | 커스텀 (비공개) | 추가 학습 데이터 |

- 컬럼 구조: `code` (취약 코드) + `desc` (취약점 설명) → SFT 포맷 변환
- `llama-model/data/` 전체 `.gitignore` 처리 — 팀 공유 시 별도 전달

### 평가 데이터

| 데이터셋 | HuggingFace 링크 |
|---------|-----------------|
| vulnerable-code | [`doss1232/vulnerable-code`](https://huggingface.co/datasets/doss1232/vulnerable-code) |

### CVE 데이터

NVD API에서 수집 (`CVE/cve_downloader.py`):

- 출처: [nvd.nist.gov/developers/vulnerabilities](https://nvd.nist.gov/developers/vulnerabilities)
- 생성 파일: `CVE/cve_database.txt` → `cve_index.faiss` + `cve_data.pkl`
- 모두 `.gitignore` 처리 (용량 문제)

---

## 폴더 구조

```
llama-guard/
├── CVE/                            # CVE 데이터 관련
│   ├── __init__.py
│   ├── cve_downloader.py           # NVD API에서 CVE 다운로드
│   ├── cve_vectordb.py             # FAISS 벡터 DB 빌드/로드/검색
│   ├── filter_with_patches.py      # 패치 있는 CVE만 필터링
│   ├── cve_query_test.py           # CVE 검색 수동 테스트 (CVE/ 에서 실행)
│   ├── cve_index.faiss             # [생성 파일, gitignore]
│   ├── cve_data.pkl                # [생성 파일, gitignore]
│   └── cve_database*.txt           # [생성 파일, gitignore]
├── llama-model/
│   ├── README.md
│   ├── MEMO.md
│   ├── TEST.ipynb                  # 실험용 노트북
│   ├── data/                       # 학습 데이터 [gitignore]
│   │   ├── secure_programming_dpo_flat.json
│   │   └── mydata_train.json
│   ├── llama_download.py           # Hub → models/vuln_detector/ 로컬 저장
│   ├── llama_fine_tuning.py        # QLoRA SFT 학습 (GPU 필요)
│   └── llama_predict.py            # 독립 실행 추론 CLI
├── models/                         # 로컬 모델 저장소 (gitignore)
│   └── vuln_detector/              # llama_download.py 실행 결과 [gitignore]
├── workflow/
│   ├── __init__.py
│   ├── path_setup.py               # sys.path 부트스트랩 (공통 사용)
│   ├── config.py                   # 모든 설정값 중앙 관리
│   ├── graph.py                    # LangGraph 워크플로우 + CLI 진입점
│   ├── nodes.py                    # 각 노드 구현
│   ├── state.py                    # AgentState 정의 + dotenv 로드
│   └── services/
│       ├── __init__.py
│       ├── llama_service.py        # 모델 로딩/추론/CVE 검색
│       └── patch_service.py        # Upstage API (보고서 + 패치 생성)
├── docs/                           # 참조 문서 [gitignore]
├── .env                            # 실제 API 키 (절대 수정/커밋 금지)
├── .env.example                    # 환경변수 템플릿
├── pyproject.toml                  # uv 의존성 관리
└── uv.lock
```

---

## 워크플로우 실행 흐름

```
graph.py::main()
    └─ run_analysis(code)
         └─ LangGraph StateGraph.stream()
              ├─ initial_analysis_node     # LLaMA로 취약점 텍스트 분석
              ├─ detection_branch          # is_detected 기준 분기
              │    ├─ False → report_generation_node → END
              │    └─ True  → rag_node
              │                └─ cvss_calculation_node
              │                     └─ report_generation_node → END
              └─ report 필드가 최종 출력 (마크다운 문자열)
```

**진입점**: `uv run python workflow/graph.py --code "..."`

---

## 설정 (workflow/config.py)

모든 매직 넘버와 경로는 `Config` 클래스에서 관리.

| 설정 | 기본값 | 환경변수 |
|------|--------|----------|
| `MODEL_PATH` | `cycloevan/vuln_detector` | `MODEL_PATH=` |
| `HF_TOKEN` | None | `HF_TOKEN=` |
| `UPSTAGE_API_KEY` | None | `UPSTAGE_API_KEY=` |
| `SEVERITY_THRESHOLD` | `7` | — |
| `CVE_TOP_K` | `5` | — |

---

## 환경 설정

```bash
# 1. 의존성 설치 (추론 전용)
uv sync

# 2. 학습 의존성 포함
uv sync --extra train

# 3. 환경변수 설정
cp .env.example .env
# UPSTAGE_API_KEY, HF_TOKEN 입력

# 4. CVE 벡터 DB 빌드 (최초 1회)
python CVE/cve_downloader.py --output CVE/cve_database.txt
python CVE/cve_vectordb.py \
    --input CVE/cve_database.txt \
    --index-output CVE/cve_index.faiss \
    --data-output CVE/cve_data.pkl
```

`.env` 파일은 절대 수정하거나 커밋하지 않는다.

---

## 실행 예시

```bash
# SQL Injection 기본 예시
uv run python workflow/graph.py

# 직접 코드 입력
uv run python workflow/graph.py --code "import os; os.system(input())"

# 파일 분석 + 보고서 저장
uv run python workflow/graph.py --code_file path/to/code.py --output report.md

# CVE 검색 단독 테스트 (CVE/ 디렉터리에서 실행)
cd CVE && python cve_query_test.py
```

---

## 코드 수정 시 주의사항

### sys.path 관리
`workflow/` 내 파일은 `import path_setup`으로 프로젝트 루트와 `llama-model/`을 sys.path에 추가.
`workflow/services/` 하위 파일은 `path_setup` 임포트 전에 직접 `workflow/`를 sys.path에 추가해야 함 (`llama_service.py` 참조).

### 모델 로딩 파라미터
`AutoModelForCausalLM.from_pretrained()`는 반드시 `torch_dtype=` 사용 (`dtype=`은 무효).

### CVSS 임계값
`nodes.py`의 심각도 분기는 반드시 `config.SEVERITY_THRESHOLD` 참조 (하드코딩 금지).

### CVEEntry pickle 호환성
`CVE.cve_vectordb.CVEEntry`는 `graph.py`와 `nodes.py` 양쪽에서 import해야 함.
pickle은 클래스 경로가 직렬화 시점과 동일해야 역직렬화 가능하기 때문.

### Upstage API
`patch_service.py`에서 실제로 호출되는 함수는 `generate_security_report()`뿐.
`call_external_for_patch()`는 내부 헬퍼 함수로 직접 호출하지 않음.

### 학습 데이터 경로
`llama_fine_tuning.py`의 데이터 경로는 `llama-model/data/secure_programming_dpo_flat.json`.
`llama-model/data/` 디렉터리 전체가 `.gitignore` 처리됨 (`data/` 규칙).

---

## 의존성 구조

```
[core]
  torch, transformers, accelerate    ← LLaMA 추론
  faiss-cpu, sentence-transformers   ← CVE 벡터 검색
  langgraph, pydantic                ← 워크플로우 상태 관리
  openai, python-dotenv, requests    ← Upstage API / 환경변수

[train]  (uv sync --extra train)
  datasets, peft, trl, bitsandbytes  ← QLoRA 학습
  psutil, tqdm                       ← 학습 모니터링
```

---

## 알려진 구조적 제한

| 항목 | 내용 |
|------|------|
| `llama-model/` 패키지화 불가 | 디렉터리 이름에 하이픈이 포함되어 Python 패키지 불가. `sys.path` 방식 유지 |
| 학습 스크립트 OS 설정 | `llama_fine_tuning.py`에 Windows/Ubuntu 설정 주석이 혼재. GPU 환경에 맞게 수동 확인 필요 |
| `cve_query_test.py` 실행 위치 | 상대 경로를 사용하므로 반드시 `CVE/` 디렉터리에서 실행 |
| 학습 데이터 미추적 | `llama-model/data/`는 `.gitignore` 처리. 팀 공유 시 별도 전달 필요 |

# llama-model 개발 메모

## 환경 설정

프로젝트 루트에서 uv로 관리. conda 사용하지 않음.

```bash
# 학습 의존성 포함 설치 (프로젝트 루트에서)
uv sync --extra train
```

## 모델 다운로드

```bash
# HF_TOKEN을 .env에 설정한 뒤
HF_TOKEN=your_token python llama-model/llama_download.py
```

저장 위치: `models/vuln_detector/`

## 학습 실행

```bash
# llama-model/ 내에 base 모델이 있어야 함 (llama-3.2-1B-Instruct/)
uv run python llama-model/llama_fine_tuning.py
```

출력: `llama-model/merged-vuln-detector/` (병합된 전체 모델)

## 독립 추론 CLI

```bash
uv run python llama-model/llama_predict.py \
  --model cycloevan/vuln_detector \
  --code "def login(u, p): ..."
```

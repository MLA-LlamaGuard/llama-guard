### Llama Model Tuning

### Model

- model : meta-llama/Llama-3.2-1B-Instruct
- Training Method : QLoRA
- 실행환경 : window / python


### TODOLIST
- Input / Output 확인
  > Input : Code (함수 단위) / Output : Vuln_Description 
  > Vuln_Description : 취약점 설명 처럼 나오면 좋음
- 데이터를 어떻게 학습시켜야하는지 
    (적합한 오픈 데이터 확인)
- 학습 및 추론 방법






### Dataset
[CyberNative/Code_Vulnerability_Security_DPO](https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO)





- 프로젝트 계획서
```
팀 정보
    팀명: 라마 가드 (LLaMA Guard)
    팀원 정보
    팀장: TBD 
팀원:  원인영, 현규원, 정석희, 백승윤, 우지수

프로젝트 정보
- 프로젝트 주제
  코드 리뷰를 통한 자동 코드 취약점 분석 시스템

- 프로젝트 세부 내용
  이미 알려진 취약점과 그 예제 코드를 LLM에게 in-context learning이나 fine-tuning등을 통해 미리 학습시킴
  리뷰할 코드가 발생되면 미리 준비된 LLM을 통해 취약점이 발생할 가능성을 분석
  분석 내용의 정확도를 feedback등을 통해  상승시큼

- AI 활용 기술
  fine-tuning내지는 in-context 러닝: AI에게 취약한 코드의 설명 및 예시를 pair로 학습시켜서 보안에 입각한 코드 리뷰에 특화된 결과물을 내도록 유도
  - LLM에게 이 코드가 왜 위험한지를 자연어로 설명시키고, 코드 대안을 제시하도록 학습시킴2
```


- 프로젝트 구성도 가안
```MERMAID
graph TD
    A[코드 입력] --> B[Code Analysis Agent]
    B --> C{위험도 > 0.7?}
    C -->|Yes| D[Vector Search Agent]
    C -->|No| E[Report Generation Agent]
    D --> E
    E --> F[최종 취약점 리포트]

    G[LLaMA-1B-Instruct] --> B
    H[FAISS Vector DB] --> D
    I[Upstage API] --> E
```
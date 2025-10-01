import json, argparse

def load_any(path):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    # 단일 JSON 시도
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            if "data" in obj and isinstance(obj["data"], list):
                return obj["data"]         # 이미 공식 포맷
            else:
                # 혹시 평탄화 리스트가 dict로 들어있으면 감지 실패 가능
                pass
        if isinstance(obj, list):
            return obj                     # HF 평탄화 리스트
    except json.JSONDecodeError:
        pass
    # line 단위(JSONL) 처리
    items = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except:
            continue
        if isinstance(o, dict) and "data" in o:
            # 공식 조각
            return o["data"]
        items.append(o)
    return items

def to_official(data_list):
    """
    입력: HF 평탄화 예시
      {"id":..., "context":..., "question":..., "answers":{"text":[...], "answer_start":[...]}, "title":...}
    출력: KorQuAD 공식 계층형
    """
    # context 기준으로 묶기
    buckets = {}  # (title, context) -> list of qas
    for ex in data_list:
        if isinstance(ex, dict) and "paragraphs" in ex:
            # 이미 공식 포맷이면 그대로 반환
            return {"version": "KorQuAD_v1.0_dev_made", "data": data_list}
        qid = ex["id"]
        ctx = ex["context"]
        q   = ex["question"]
        ans = ex.get("answers", {})
        texts = ans.get("text", [])
        starts = ans.get("answer_start", [])
        title = ex.get("title", "")
        key = (title, ctx)
        qa = {
            "id": qid,
            "question": q,
            "answers": [
                {"text": t, "answer_start": starts[i] if i < len(starts) else -1}
                for i, t in enumerate(texts)
            ],
            "is_impossible": False
        }
        buckets.setdefault(key, []).append(qa)

    articles = []
    # title 단위로 article 구성
    from collections import defaultdict
    by_title = defaultdict(list)  # title -> list of (context, qas)
    for (title, ctx), qas in buckets.items():
        by_title[title].append({"context": ctx, "qas": qas})

    for title, paras in by_title.items():
        articles.append({
            "title": title,
            "paragraphs": paras
        })

    return {"version": "KorQuAD_v1.0_dev_made", "data": articles}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    args = ap.parse_args()

    data_list = load_any(args.infile)
    official = to_official(data_list)
    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(official, f, ensure_ascii=False)
    print("wrote:", args.outfile)

import json

# 저장된 preds 파일 로드
with open("preds_base.json", "r", encoding="utf-8") as f:
    bases = json.load(f)
with open("preds_lora.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

print("총 예측 개수:", len(preds))

with open("./KorQuAD_v1.0/korquad_val_flatten.json", "r", encoding="utf-8") as f:
    dev = json.load(f)["data"]

# 확인하고 싶은 qid 리스트
target_qids = {
    "6260132-4-0",
    "6559611-9-1",
    "6479772-0-1",
    "6550021-12-0",
    "6545182-21-1",
    "6533020-1-1",
    "6517222-4-0",
    "6543817-1-0",
    "6657570-21-1",
    "5840144-8-2",
}

count = 0
for article in dev:
    for para in article["paragraphs"]:
        for qa in para["qas"]:
            qid = qa["id"]
            if qid in target_qids and qid in preds and qid in bases:
                base_ans = bases[qid].strip()
                lora_ans = preds[qid].strip()
                #print("qid       :", qid)
                print("질문      :", qa["question"])
                print("Base 예측 :", base_ans)
                print("LoRA 예측 :", lora_ans)
                print("정답      :", qa["answers"][0]["text"])
                print("-" * 40)
                count += 1
        if count >= len(target_qids):
            break
    if count >= len(target_qids):
        break

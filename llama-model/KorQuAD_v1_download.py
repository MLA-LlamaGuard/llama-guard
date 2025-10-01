import random
import math
from datasets import load_dataset
import os

dataset = load_dataset("KorQuAD/squad_kor_v1")

train_dataset = dataset["train"]
val_dataset = dataset["validation"]

# 로컬 저장 경로
base_dir = "./KorQuAD_v1.0"
os.makedirs(base_dir, exist_ok=True)
train_path = os.path.join(base_dir, "korquad_train.json")
val_path = os.path.join(base_dir, "korquad_original_validation.json")

train_dataset.to_json(train_path, force_ascii=False)
val_dataset.to_json(val_path, force_ascii=False)

# 반드시 포함해야 하는 qid들
required_ids = {
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
# 재현성 고정
random.seed(42)

# 타깃 사이즈: validation의 1/4 (내림)
val_size = len(val_dataset)
target_size = max(1, val_size // 4)

# id -> 인덱스 매핑
# KorQuAD v1 스킴: 각 예제에 'id' 필드가 있음
id2idx = {}
for i in range(val_size):
    ex_id = val_dataset[i]["id"]
    # 혹시 중복 id가 있다면 첫 인덱스만 보존
    if ex_id not in id2idx:
        id2idx[ex_id] = i

# 실제로 존재하는 required id만 추림
present_required_ids = [qid for qid in required_ids if qid in id2idx]
missing_required_ids = [qid for qid in required_ids if qid not in id2idx]

if missing_required_ids:
    print("[경고] validation에서 찾을 수 없는 qid가 있습니다:", missing_required_ids)

# 우선 필수 샘플 인덱스 집합
selected_indices = {id2idx[qid] for qid in present_required_ids}

# 남은 필요 개수 계산
remaining_needed = max(0, target_size - len(selected_indices))

# 남은 후보 인덱스 (필수 제외)
remaining_pool = [i for i in range(val_size) if i not in selected_indices]

# 풀에서 랜덤 샘플링하여 채움
if remaining_needed > len(remaining_pool):
    # 이 경우는 거의 없겠지만, 풀보다 많이 필요하면 가능한 만큼만 사용
    print("[주의] 남은 후보 수가 부족하여 타깃 사이즈로 정확히 맞출 수 없습니다.")
    remaining_needed = len(remaining_pool)

extra_indices = set(random.sample(remaining_pool, remaining_needed))
selected_indices |= extra_indices

# 최종 선택 인덱스 리스트(정렬하면 재현성/가독성에 도움)
final_indices = sorted(selected_indices)

# 부분 데이터셋 생성
val_quarter = val_dataset.select(final_indices)

# 저장
val_quarter_path = os.path.join(base_dir, "korquad_validation.json")
val_quarter.to_json(val_quarter_path, force_ascii=False)

# 리포트
print(dataset)

print("\nTrain 개수:", len(train_dataset))
print("원본 Validation 개수:", len(val_dataset))
print("사용할 Validation 개수:", len(val_quarter))

print("\nTrain example:\n", train_dataset[0])
print("\nValidation example:\n", val_dataset[0])

print("\nTrain 파일 크기: %.2f MB" % (os.path.getsize(train_path)/(1024*1024)))
print("원본 Validation 파일 크기: %.2f MB" % (os.path.getsize(val_path)/(1024*1024)))
print("사용할 Validation 파일 크기: %.2f MB" % (os.path.getsize(val_quarter_path)/(1024*1024)))

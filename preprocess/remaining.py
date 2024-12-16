import json

# 设置你的文件名
original_file = '/home/disk/huanghui/data/judgelm/judgelm_train_100k.jsonl'
extracted_file = '/home/disk/huanghui/data/judgelm/judgelm_sampled_data.jsonl'
remaining_file = '/home/disk/huanghui/data/judgelm/judgelm_remaining_data.jsonl'

# 读取已经提取了 2 万条数据的 jsonl 文件
extracted_ids = set()
with open(extracted_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        extracted_ids.add(data["review_id"])

# 处理原始文件，并将那些不在原文件中的条目保存到新文件
with open(original_file, 'r') as f_original, open(remaining_file, 'w') as f_remaining:
    for line in f_original:
        data = json.loads(line)
        if data["review_id"] not in extracted_ids:
            f_remaining.write(json.dumps(data) + '\n')

print(f'Remaining data has been saved to {remaining_file}.')
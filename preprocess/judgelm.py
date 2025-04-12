import json

input_filename = 'data/judgelm/judgelm_sampled_data.jsonl'
output_filename = 'data/judgelm/judgelm.json'

# 初始化列表以存储处理后的数据
processed_data = []

# 读取 jsonl 文件并处理每一行
with open(input_filename, 'r', encoding='utf-8') as input_file:
    for line in input_file:
        # 解析 JSON 数据
        data = json.loads(line)
        # 获取两个分数和对应的答案
        scores = data['score']
        answer_bodies = [data['answer1_body'], data['answer2_body']]
        # 确定得分更高和更低的答案
        max_index = scores.index(max(scores))
        min_index = 1 - max_index
        # 创建包含问题、最佳答案和最差答案的字典
        processed_entry = {
            'input': data['question_body'],
            'better': answer_bodies[max_index],
            'worse': answer_bodies[min_index]
        }
        # 将处理后的条目添加到列表中
        processed_data.append(processed_entry)

# 将处理后的数据列表保存到 JSON 文件中
with open(output_filename, 'w', encoding='utf-8') as output_file:
    json.dump(processed_data, output_file, indent=4, ensure_ascii=False)

print(f"提取了 {len(processed_data)} 条记录到 {output_filename}")

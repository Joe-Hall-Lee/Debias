import json

# 输入输出文件名
input_filename = 'data/judgelm/judgelm_sampled_data.jsonl'
output_filename = 'data/judgelm/better.json'

# 初始化一个列表用来存储每个记录的最高分 answer_body 
high_score_bodies = []

# 读取 jsonl 文件并处理每一行
with open(input_filename, 'r') as input_file:
    for line in input_file:
        # 解析 JSON 数据
        data = json.loads(line)
        # 获取两个分数和对应的答案
        scores = data['score']
        answer_bodies = [data['answer1_body'], data['answer2_body']]
        # 确定得分更高的答案
        high_score_body = answer_bodies[scores.index(max(scores))]
        # 将得分最高的 answer_body 加入列表
        high_score_bodies.append({'input': data['question_body'],'output': high_score_body})

# 将最高分答案的列表保存到 json 文件中
with open(output_filename, 'w') as output_file:
    json.dump(high_score_bodies, output_file, indent=4)

print(f"Extracted {len(high_score_bodies)} high score answers to {output_filename}")
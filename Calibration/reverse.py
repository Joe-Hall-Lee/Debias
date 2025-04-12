import json


# 用于按位计算两个列表的差值，并返回结果列表
def compute_difference(list_a, list_b):
    return [a - b for a, b in zip(list_a, list_b)]


def process_files(file1_path, file2_path, file3_path, output_path):
    count = 0
    with open(file1_path, 'r', encoding='utf-8') as file1, \
            open(file2_path, 'r', encoding='utf-8') as file2, \
            open(file3_path, 'r', encoding='utf-8') as file3, \
            open(output_path, 'w', encoding='utf-8') as output_file:
        for i, (line1, line2, line3) in enumerate(zip(file1, file2, file3), 1):
            try:
                data1 = json.loads(line1)
                data2 = json.loads(line2)
                data3 = json.loads(line3)
                count += 1
            except json.JSONDecodeError:
                print(f"JSONDecodeError at line {i} in file3: {line3}")
                continue

            # 从 pred_text 获取数字并以空格分割它们
            nums1 = [float(num) for num in data1['pred_text'].split()]
            nums2 = [float(num) for num in data2['pred_text'].split()]
            nums3 = [float(num) for num in data3['pred_text'].split()]

            # 计算第二个和第三个 pred_text 值的差
            primary_diff = compute_difference(nums2, nums3)

            # 把差值的两个数值交换位置
            swapped_diff = primary_diff[::-1]

            # 计算第一个文件的 pred_text 与交换后的结果相减
            final_result = compute_difference(nums1, swapped_diff)

            # 更新 data1 的 pred_text 为最终结果，并写入输出文件中
            data1['pred_text'] = ' '.join(map(str, final_result))
            json.dump(data1, output_file, ensure_ascii=False)
            output_file.write('\n')


process_files('F:/CS/AI/LLM-Evaluation/GPT4/Vanilla_CoT/judgements_output/Adversarial/Neighbor/7b-full-model-reverse',
              'F:/CS/AI/LLM-Evaluation/GPT4/Vanilla_CoT/judgements_output/Adversarial/Neighbor/7b-full-model',
              'F:/CS/AI/LLM-Evaluation/GPT4/Superficial_CoT/judgements_output/Adversarial/Neighbor/7b-full-model',
              'F:/CS/AI/LLM-Evaluation/GPT4/Superficial_CoT/judgements_output/Adversarial/Neighbor/7b-full-model-reverse')

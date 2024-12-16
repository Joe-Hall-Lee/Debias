from judgelm.utils import extract_jsonl
from judgelm.llm_judge.common import load_questions, reorg_answer_file, conv_judge_pair, parse_score, \
    translate_score_to_win_list
import argparse
import os
import time
import json
import requests
import torch
from tqdm import tqdm

import sys
from pathlib import Path
import traceback
from request_tokens import request_tokens
file = Path(__file__).resolve()
root = file.parents[2]
sys.path.append(str(root))
print(sys.path)


def run_eval(
        question_file,
        vanilla_file,
        answer_file,
        if_reverse_answers,
        if_fast_eval,
):
    print("start run_eval")
    questions = load_questions(question_file)

    get_answers_func = get_model_answers

    chunk_size = len(questions)
    ans_handles = []
    print("start ans_handles append")
    for i in range(86, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                questions[i: i + chunk_size],
                vanilla_file,
                answer_file,
                if_reverse_answers,
                if_fast_eval,
            )
        )


# Add a global list to keep track of errors
errors_list = []


@torch.inference_mode()
def get_model_answers(
        questions,
        vanilla_file,
        answer_file,
        if_reverse_answers,
        if_fast_eval,
):
    # 使用 extract_jsonl 函数提取并加载预测数据，构建一个便于查询的字典
    pred_data = {item['question_id']: item['pred_text']
                 for item in extract_jsonl(vanilla_file)}

    for q_i, question in tqdm(enumerate(questions)):
        try:
            torch.manual_seed(q_i)
            conv = conv_judge_pair.copy()

            # if fast eval, use the "\n" as the separator
            if if_fast_eval:
                conv.sep = "\n"

            # reverse the order of the answers
            if if_reverse_answers:
                temp_answer = question["answer1_body"]
                question["answer1_body"] = question["answer2_body"]
                question["answer2_body"] = temp_answer

            # prompt1
            system_superficial = "You are a meticulous evaluator whose task is to assess the superficial quality of an AI assistant's response, and you should focus specifically on language expression without considering the factual accuracy of the information provided."

            prompt_superficial = "Evaluate the superficial quality of the provided answer in terms of linguistic expression and stylistic presentation. Provide a score between 1 and 10, where 10 signifies exceptional superficial articulation encompassing aspects such as lexical diversity, structural coherence, stylistic elegance, and overall fluidity.\n  On the first line, offer a detailed rationale for your score, explaining how well the answer demonstrates each assessed quality aspect. Your analysis should be thorough and impartial, focusing solely on superficial elements.\n  On the subsequent line, your rating should be presented as a numerical value without any other comments or explanations. There should be nothing on this line except a score."

            prompt_template_superficial = "[The Start of Answer]\n{answer}\n\n[The End of Answer]\n\n[System]\n{prompt}\n\n"
            data_sample_superficial_1 = system_superficial + '\n' + prompt_template_superficial.format(
                answer=question['answer1_body'], prompt=prompt_superficial) + conv.appendix

            data_sample_superficial_2 = system_superficial + '\n' + prompt_template_superficial.format(
                answer=question['answer2_body'], prompt=prompt_superficial) + conv.appendix

            # generate judgements
            # 从预加载的数据中获取预测分数
            question_id = question['question_id']

            output = pred_data.get(question_id, 'No data found')

            output_superficial_1 = request_tokens(data_sample_superficial_1)
            output_superficial_2 = request_tokens(data_sample_superficial_2)

            # 分割输出以分离评估和分数
            parts = output_superficial_1.split('\n')
            output_superficial_1 = parts[-1]

            parts = output_superficial_2.split('\n')
            output_superficial_2 = parts[-1]

            output = output.strip().split()
            output_superficial_1 = output_superficial_1.strip()
            output_superficial_2 = output_superficial_2.strip()

            output_score_1, output_score_2 = 0.0, 0.0
            try:
                output_score_1 = float(
                    output[0]) - 0.8 * float(output_superficial_1)
                output_score_2 = float(
                    output[1]) - 0.8 * float(output_superficial_2)
            except Exception as score_error:
                output_score = "Error calculating scores"
                # Add the question_id to the errors list
                errors_list.append(question_id)
                print(
                    f"Score calculation error for question_id {question_id}: {score_error}")

            # Only proceed to write output if scores were calculated successfully
            if output_score != "Error calculating scores":
                output_score = str(output_score_1) + " " + str(output_score_2)

                # Dump answers
                os.makedirs(os.path.dirname(answer_file), exist_ok=True)
                with open(os.path.expanduser(answer_file), "a") as fout:
                    question["pred_text"] = output_score
                    fout.write(json.dumps(question) + "\n")

        except Exception as e:
            # Record the entire traceback to identify where exactly things went wrong
            error_msg = traceback.format_exc()
            errors_list.append(question_id)
            print(
                f"Unexpected error for question_id {question_id}: {error_msg}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--question-file",
        type=str,
        default="F:/CS/AI/LLM-Evaluation/Dataset/LLMBar/Adversarial/GPTInst/converted_dataset.jsonl",
        help="The name of the benchmark question set.",
    )

    parser.add_argument(
        "--answer-file",
        type=str,
        default="F:/CS/AI/LLM-Evaluation/GPT4/Superficial_CoT/judgements_output/Adversarial/GPTInst/7b-full-model",
        help="The output answer file.")

    parser.add_argument(
        "--vanilla-file",
        type=str,
        default="F:/CS/AI/LLM-Evaluation/GPT4/Vanilla_CoT/judgements_output/Adversarial/GPTInst/7b-full-model",
        help="The GPT4-vanilla output answer file.")

    parser.add_argument(
        "--if-reverse-answers",
        type=int,
        default=0,
        help="Whether to reverse the order of the answers.",
    )

    parser.add_argument(
        "--if-fast-eval",
        type=int,
        default=1,
        help="Whether to use fast evaluation.",
    )

    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    args = parser.parse_args()
    args.if_reverse_answers = bool(args.if_reverse_answers)
    args.if_fast_eval = bool(args.if_fast_eval)

    print(f"args: {args}")

    print(f"Output to {args.answer_file}")

    run_eval(
        args.question_file,
        args.vanilla_file,
        args.answer_file,
        args.if_reverse_answers,
        args.if_fast_eval
    )

    reorg_answer_file(args.answer_file)

    # statistics the judgements
    sequential_pred_answer_file_list = extract_jsonl(args.answer_file)

    sequential_pred_score_list = []
    for sequential_pred_answer_file in sequential_pred_answer_file_list:
        sequential_pred_score_list.append(parse_score(
            sequential_pred_answer_file['pred_text']))

    # if the score gap is less than T, we consider it as a draw
    T = 0.0
    sequential_pred_win_list = translate_score_to_win_list(
        sequential_pred_score_list, T)

    # get the number of 1 in sequential_pred_win_list
    win_num = sequential_pred_win_list.count(1)
    tie_num = sequential_pred_win_list.count(0)
    lose_num = sequential_pred_win_list.count(-1)

    # print the win, tie, and lose number, use format {}
    print("Assistant 1's reuslts ---> win_num: {}, tie_num: {}, lose_num: {}".format(win_num, tie_num, lose_num))
    print("Assistant 2's reuslts ---> win_num: {}, tie_num: {}, lose_num: {}".format(lose_num, tie_num, win_num))

    # At the end of execution, output the questions that had errors
    if errors_list:
        print("The following question_ids had errors:")
        for err_id in errors_list:
            print(err_id)

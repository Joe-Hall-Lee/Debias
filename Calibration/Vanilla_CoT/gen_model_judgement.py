from judgelm.utils import extract_jsonl
from judgelm.llm_judge.common import load_questions, reorg_answer_file, conv_judge_pair, parse_score, translate_score_to_win_list
import argparse
import json
import os
import json
from request_tokens import request_tokens
import torch
from tqdm import tqdm

import sys
from pathlib import Path
file = Path(__file__).resolve()
root = file.parents[2]
sys.path.append(str(root))
print(sys.path)


def run_eval(
    question_file,
    answer_file,
    if_reverse_answers,
    if_fast_eval
):
    print("start run_eval")
    questions = load_questions(question_file)

    get_answers_func = get_model_answers

    chunk_size = len(questions)
    ans_handles = []
    print("start ans_handles append")
    for i in range(93, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                questions[i: i + chunk_size],
                answer_file,
                if_reverse_answers,
                if_fast_eval,
            )
        )


@torch.inference_mode()
def get_model_answers(
    questions,
    answer_file,
    if_reverse_answers,
    if_fast_eval,
):

    for q_i, question in tqdm(enumerate(questions)):
        torch.manual_seed(q_i)
        conv = conv_judge_pair.copy()
        template = conv.prompt_template

        # if fast eval, use the "\n" as the separator
        if if_fast_eval:
            conv.sep = "\n"

        # reverse the order of the answers
        if if_reverse_answers:
            temp_answer = question["answer1_body"]
            question["answer1_body"] = question["answer2_body"]
            question["answer2_body"] = temp_answer

        # combine data_sample

        data_sample = conv.system + '\n' + template.format(question=question['question_body'],
                                                           answer_1=question['answer1_body'],
                                                           answer_2=question['answer2_body'],
                                                           prompt=conv.prompt) + conv.appendix

        # generate judgements
        output = request_tokens(data_sample)

        # 分割输出以分离评估和分数
        parts = output.split('\n')
        output = parts[-1] if len(parts) > 1 else None
        # print(output)

        output = output.strip()

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            question["pred_text"] = output
            fout.write(json.dumps(question) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--question-file",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str,
                        help="The output answer file.")

    parser.add_argument(
        "--if-reverse-answers",
        type=int,
        default=0,
        help="Whether to reverse the order of the answers.",
    )

    parser.add_argument(
        "--if-fast-eval",
        type=int,
        default=0,
        help="Whether to use fast evaluation.",
    )
    args = parser.parse_args()
    args.if_reverse_answers = bool(args.if_reverse_answers)
    args.if_fast_eval = bool(args.if_fast_eval)

    print(f"args: {args}")

    print(f"Output to {args.answer_file}")

    run_eval(
        args.question_file,
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

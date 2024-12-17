from dataclasses import dataclass, field
import json
import copy
import pathlib
from typing import Dict

import numpy as np
import torch
from typing import Dict, Optional
import transformers
from transformers import Trainer, AutoModelForCausalLM


from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_type: str = field(default="llama")
    class_type: str = field(default="generation")
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = True
    swap_aug_ratio: float = 0
    ref_drop_ratio: float = 1


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def create_prompt():
    instruction = {}
    instruction["noref"] = "You are a helpful and precise assistant for checking the quality of the answer.\n[Question]\n{question_body}\n\n[The Start of Assistant 1's Answer]{answer1_body}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer2_body}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{rubric}\n\n### Response:"
    return instruction


def swap_first_two_integers(s):
    # find the first space
    first_space_index = s.find(' ')
    if first_space_index != -1:
        # find the second space
        second_space_index = s.find('\n', first_space_index + 1)
        if second_space_index != -1:
            # swap the first two integers
            new_s = s[first_space_index + 1:second_space_index] + ' ' + \
                s[:first_space_index] + '\n' + s[second_space_index + 1:]
            return new_s
    return s


def format_instruction(instruction, example, class_type):
    if "rubric" not in example.keys():
        example["rubric"] = "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."
    if "reference" not in example.keys():
        example["reference"] = {"text": None}
    if class_type in ["multi-regression", "classification", "multi-contrast", "generation"]:
        prompt = instruction.format(question_body=example["question_body"],
                                    rubric=example["rubric"],
                                    reference=example["reference"]['text'],
                                    answer1_body=example["answer1_body"],
                                    answer2_body=example["answer2_body"])
        return prompt


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    class_type,
    instruction,
    swap_aug_ratio,
    ref_drop_ratio,
) -> Dict:

    # Apply prompt templates
    conversations = []
    labels = []
    for i, source in enumerate(sources):
        if ref_drop_ratio > 0 and np.random.rand() < ref_drop_ratio:
            instruction = instruction["noref"]
        else:
            instruction = instruction["ref"]
            source["score"] = source["score_w_reference"]
            source["text"] = source["text_w_reference"]

        if swap_aug_ratio > 0 and np.random.rand() < swap_aug_ratio:
            if "answer2_body" in source.keys():
                source["answer2_body"], source["answer1_body"] = source["answer1_body"], source["answer2_body"]
                source['score'] = [source['score'][1], source['score'][0]]
                source['text'] = swap_first_two_integers(source['text'])
                source['text'] = source['text'].replace(
                    'Assistant 1', 'Assistant X')
                source['text'] = source['text'].replace(
                    'Assistant 2', 'Assistant 1')
                source['text'] = source['text'].replace(
                    'Assistant X', 'Assistant 2')

        if "rubric" not in source.keys():
            source["rubric"] = "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."

        if class_type == "generation":
            prompt = format_instruction(instruction, source, class_type)
            conversations.append(prompt)
            labels.append(source['text'])

    return conversations, labels


class LazySupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, instruction, class_type, ref_drop_ratio, swap_aug_ratio):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.instruction = instruction
        self.class_type = class_type
        self.ref_drop_ratio = ref_drop_ratio
        self.swap_aug_ratio = swap_aug_ratio


class GenerationLazySupervisedDataset(LazySupervisedDataset):
    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        conversations, labels = preprocess(
            [self.raw_data[i]],
            self.tokenizer,
            self.class_type,
            self.instruction,
            self.swap_aug_ratio,
            self.ref_drop_ratio,
        )
        conv_labels = [conversations[0] + labels[0]]
        # Tokenize conversations
        tokenized = self.tokenizer(
            conv_labels,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        labels = copy.deepcopy(tokenized.input_ids)

        ret = dict(
            input_ids=tokenized.input_ids[0],
            labels=labels[0],
            attention_mask=tokenized.attention_mask[0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, class_type, instruction, ref_drop_ratio, swap_aug_ratio,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if class_type == "generation":
        dataset_cls = GenerationLazySupervisedDataset

    rank0_print("Loading data...")

    with open(data_args.data_path, "r") as fin:
        train_data = [json.loads(line) for line in fin.readlines()]

    train_dataset = dataset_cls(train_data,
                                tokenizer=tokenizer,
                                instruction=instruction,
                                class_type=class_type,
                                ref_drop_ratio=ref_drop_ratio,
                                swap_aug_ratio=swap_aug_ratio,
                                )

    rank0_print("Loading data finished")

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    config.use_cache = False

    if model_args.class_type == "generation":
        MODEL_CLS = AutoModelForCausalLM

    model = MODEL_CLS.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=True,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )

    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    instruction = create_prompt()

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        class_type=model_args.class_type,
        instruction=instruction,
        ref_drop_ratio=data_args.ref_drop_ratio,
        swap_aug_ratio=data_args.swap_aug_ratio,
    )

    # Start trainer
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            trainer.model, StateDictType.FULL_STATE_DICT, save_policy
        ):
            trainer.save_model()


if __name__ == "__main__":
    train()

import re
import argparse
import numpy as np

from unsloth import FastLanguageModel
from datasets import load_dataset, interleave_datasets, concatenate_datasets
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from utils import load_config, save_config_to_output_dir, DatasetConfig


INSTRUCTION_PROMPT = """{question}
### 정답: {response}"""

MCQA_PROMPT_REASONING = """### 질문: {question}
### 선택지:
{choices}
### 정답: {cot}"""

MCQA_PROMPT_FINAL = """### 질문: {question}
### 선택지:
{choices}
### 정답: {cot}
### 정답: {answer}"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the YAML configuration file",
)
args = parser.parse_args()

config = load_config(args.config)
config.model.dtype = None if config.model.dtype is None else config.model.dtype

# config 저장
output_dir = (
    config.training.output_dir
)  # Use the output directory from the configuration
save_config_to_output_dir(config, output_dir)

# Load Model and Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config.model.name,
    max_seq_length=config.model.max_seq_length,
    dtype=config.model.dtype,
    load_in_4bit=config.model.load_in_4bit,
)

EOS_TOKEN = tokenizer.eos_token

# LoRA Adapter
model = FastLanguageModel.get_peft_model(
    model,
    r=config.lora.r,
    target_modules=config.lora.target_modules,
    lora_alpha=config.lora.lora_alpha,
    lora_dropout=config.lora.lora_dropout,
    bias=config.lora.bias,
    use_gradient_checkpointing=config.lora.use_gradient_checkpointing,
    random_state=config.lora.random_state,
    use_rslora=config.lora.use_rslora,
    loftq_config=config.lora.loftq_config,
)


def contains_foreign_language(text):
    # 정규식 패턴: 영어, 한국어 문자를 제외한 다양한 언어의 유니코드 범위
    pattern = r"[^\u0000-\u007F\uAC00-\uD7AF]"  # 영어 (U+0000-U+007F) 및 한국어 (U+AC00-U+D7AF) 범위를 제외한 문자

    # 유니코드 범위를 벗어난 문자가 포함된 경우 찾기
    if re.search(pattern, text):
        return True
    return False


def instruction_filter_func(example):
    # Filter out rows where the output contains "죄송" or foreign languages
    if "죄송" in example["response"]:
        return False
    if contains_foreign_language(example["response"]):
        return False
    return True


def instruction_formatting_prompts_func(examples):
    instructions = examples["prompt"]
    outputs = examples["response"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        # text = instruction + output + EOS_TOKEN # 마지막에 eos token을 추가해줌으로써 모델이 출력을 끝마칠 수 있게 만들어 줍니다.
        text = (
            INSTRUCTION_PROMPT.format(question=instruction, response=output) + EOS_TOKEN
        )  # 마지막에 eos token을 추가해줌으로써 모델이 출력을 끝마칠 수 있게 만들어 줍니다.
        texts.append(text)
    return {
        "formatted_text": texts,
    }


def mcqa_filter_func(example):
    choices = example["choices"]
    answer_idx = example["answer"]

    if any(re.match("^[A-Z].", choice) is None for choice in choices):
        return False
    check_choices = [re.sub("^[A-Z].", "", choice).strip() for choice in choices]
    if len(set(check_choices)) != len(check_choices):
        return False
    if answer_idx >= len(choices):
        return False
    return True


def mcqa_formatting_prompts_func(examples):
    question = examples["question"][0]
    answer_id = examples["answer"][0]
    choice = examples["choices"][0]
    cot = examples["cot"][0]

    answer = choice[answer_id]
    answer_alphabet = answer[0]
    choice_str = "\n".join(choice)

    mcqa_reasoning = (
        MCQA_PROMPT_REASONING.format(question=question, choices=choice_str, cot=cot)
        + EOS_TOKEN
    )  # 마지막에 eos token을 추가해줌으로써 모델이 출력을 끝마칠 수 있게 만들어 줍니다.
    mcqa_final = (
        MCQA_PROMPT_FINAL.format(
            question=question,
            choices=choice_str,
            cot=cot,
            answer=answer_alphabet,
        )
        + EOS_TOKEN
    )  # 마지막에 eos token을 추가해줌으로써 모델이 출력을 끝마칠 수 있게 만들어 줍니다.

    return {"formatted_text": [mcqa_reasoning, mcqa_final]}


def preprocess_instruction_dataset(dataset):
    dataset = dataset.filter(instruction_filter_func)
    dataset = dataset.map(
        instruction_formatting_prompts_func,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return dataset


def preprocess_mcqa_dataset(dataset):
    dataset = dataset.filter(mcqa_filter_func)
    dataset = dataset.map(
        mcqa_formatting_prompts_func,
        batched=True,
        batch_size=1,
        remove_columns=dataset.column_names,
    )
    return dataset


train_datasets = []
valid_datasets = []
ratios = []

for dataset_config in config.datasets.get("instruction", []):
    dataset_config: DatasetConfig
    dataset = load_dataset(dataset_config.name, split="train")
    train_test_split = dataset.train_test_split(test_size=100, seed=42)
    train_dataset = preprocess_instruction_dataset(train_test_split["train"])
    valid_dataset = preprocess_instruction_dataset(train_test_split["test"])
    train_datasets.append(train_dataset)
    valid_datasets.append(valid_dataset)
    ratios.append(dataset_config.ratio)

for dataset_config in config.datasets.get("mcqa", []):
    dataset_config: DatasetConfig
    dataset = load_dataset(dataset_config.name, split="train")
    train_test_split = dataset.train_test_split(test_size=100, seed=42)
    train_dataset = preprocess_mcqa_dataset(train_test_split["train"])
    valid_dataset = preprocess_mcqa_dataset(train_test_split["test"])
    train_datasets.append(train_dataset)
    valid_datasets.append(valid_dataset)
    ratios.append(dataset_config.ratio)

ratios = np.array(ratios)
probabilities = ratios / np.sum(ratios)

# Combine training datasets using interleave_datasets
combined_train_dataset = interleave_datasets(
    train_datasets, probabilities=probabilities
)
combined_train_dataset = combined_train_dataset.shuffle(seed=42)

# Combine validation datasets (equal weighting since validation is independent of sampling ratios)
combined_valid_dataset = concatenate_datasets(valid_datasets)
combined_valid_dataset = combined_valid_dataset.shuffle(seed=42)

data_collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer, mlm=False, response_template="### 정답: "
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=combined_train_dataset,
    eval_dataset=combined_valid_dataset,
    dataset_text_field="formatted_text",
    max_seq_length=config.model.max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        **vars(config.training),
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
    ),
)

trainer_stats = trainer.train()

# LoRA Adapter 저장
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Merged model 저장 및 업로드
model.save_pretrained_merged(
    "model",
    tokenizer,
    save_method="merged_16bit",
)

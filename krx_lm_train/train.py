import re

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported


max_seq_length = 2048
dtype = None # None으로 지정할 경우 해당 컴퓨팅 유닛에 알맞은 dtype으로 저장됩니다. Tesla T4와 V100의 경우에는 Float16, Ampere+ 이상의 경우에는 Bfloat16으로 설정됩니다.
load_in_4bit = True # 메모리 사용량을 줄이기 위해서는 4bit 양자화를 사용하실 것을 권장합니다.

# 모델 및 토크나이저 선언
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2-7B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # gated model을 사용할 경우 허깅페이스 토큰을 입력해주시길 바라겠습니다.
)

# LoRA Adapter 선언
model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # 0을 넘는 숫자를 선택하세요. 8, 16, 32, 64, 128이 추천됩니다.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",], # target module도 적절하게 조정할 수 있습니다.
    lora_alpha = 16,
    lora_dropout = 0, # 어떤 값이든 사용될 수 있지만, 0으로 최적화되어 있습니다.
    bias = "none",    # 어떤 값이든 사용될 수 있지만, "none"으로 최적화되어 있습니다.
    use_gradient_checkpointing = "unsloth", # 매우 긴 context에 대해 True 또는 "unsloth"를 사용하십시오.
    random_state = 42,
    use_rslora = True,
    loftq_config = None
)

prompt_format = """다음 질문을 읽고 적절한 답변을 말하시오.
### 질문: {}
### 정답: {}"""

EOS_TOKEN = tokenizer.eos_token

def contains_foreign_language(text):
    # 정규식 패턴: 영어, 한국어 문자를 제외한 다양한 언어의 유니코드 범위
    pattern = r'[^\u0000-\u007F\uAC00-\uD7AF]'  # 영어 (U+0000-U+007F) 및 한국어 (U+AC00-U+D7AF) 범위를 제외한 문자

    # 유니코드 범위를 벗어난 문자가 포함된 경우 찾기
    if re.search(pattern, text):
        return True
    return False

def filter_func(example):
    # Filter out rows where the output contains "죄송" or foreign languages
    if "죄송" in example["response"]:
        return False
    if contains_foreign_language(example["response"]):
        return False
    return True

def formatting_prompts_func(examples):
    instructions = examples["prompt"]
    outputs = examples["response"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        #text = instruction + output + EOS_TOKEN # 마지막에 eos token을 추가해줌으로써 모델이 출력을 끝마칠 수 있게 만들어 줍니다.
        text = prompt_format.format(instruction, output) + EOS_TOKEN # 마지막에 eos token을 추가해줌으로써 모델이 출력을 끝마칠 수 있게 만들어 줍니다.
        texts.append(text)
    return { "formatted_text" : texts, }

dataset = load_dataset("amphora/krx-sample-instructions", split = "train")

# Split dataset into train and validation sets
train_test_split = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = train_test_split["train"]
valid_dataset = train_test_split["test"]

train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)
train_dataset = train_dataset.filter(filter_func)
valid_dataset = valid_dataset.map(formatting_prompts_func, batched = True,)
valid_dataset = valid_dataset.filter(filter_func)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = valid_dataset,
    dataset_text_field = "formatted_text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # True로 설정하면 짧은 텍스트 데이터에 대해서는 더 빠른 학습 속도로를 보여줍니다.
    #data_collator=data_collator,

    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.05,
        num_train_epochs = 2,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "outputs",
        save_steps = 200,
        eval_steps = 200,
        evaluation_strategy="steps",
        report_to = "none", # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()


# LoRA Adapter 저장
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Merged model 저장 및 업로드
model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
import argparse
import numpy as np

from unsloth import FastLanguageModel
from datasets import load_dataset, interleave_datasets, concatenate_datasets
from trl import CPOConfig, CPOTrainer
from unsloth import is_bfloat16_supported
from utils import load_config, save_config_to_output_dir, DatasetConfig


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

train_datasets = []
valid_datasets = []
ratios = []

for dataset_config in config.datasets.get("preference", []):
    dataset_config: DatasetConfig
    dataset = load_dataset(dataset_config.name, split="train")
    train_test_split = dataset.train_test_split(test_size=100, seed=42)
    train_dataset = train_test_split["train"]
    valid_dataset = train_test_split["test"]
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


trainer = CPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=combined_train_dataset,
    eval_dataset=combined_valid_dataset,
    args=CPOConfig(
        **vars(config.training),
        max_length=config.model.max_seq_length,
        dataset_num_proc=2,
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

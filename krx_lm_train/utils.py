import yaml
import os

from dataclasses import dataclass
from typing import List, Dict, Optional, Union


@dataclass
class DatasetConfig:
    name: str
    ratio: float


@dataclass
class ModelConfig:
    name: str
    max_seq_length: int
    dtype: Optional[Union[str, None]] = None
    load_in_4bit: bool = True


@dataclass
class LoRAConfig:
    r: int
    target_modules: List[str]
    lora_alpha: int
    lora_dropout: float
    bias: str
    use_gradient_checkpointing: Union[str, bool]
    random_state: int
    use_rslora: bool
    loftq_config: Optional[Union[str, None]] = None


@dataclass
class TrainingConfig:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    num_train_epochs: int
    learning_rate: float
    logging_steps: int
    optim: str
    weight_decay: float
    lr_scheduler_type: str
    seed: int
    output_dir: str
    save_steps: int
    eval_steps: int
    evaluation_strategy: str
    report_to: Optional[str] = None


@dataclass
class Config:
    datasets: Dict[str, List[DatasetConfig]]
    model: ModelConfig
    lora: LoRAConfig
    training: TrainingConfig


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as file:
        data = yaml.safe_load(file)
    return Config(
        model=ModelConfig(**data["model"]),
        lora=LoRAConfig(**data["lora"]),
        training=TrainingConfig(**data["training"]),
        datasets={
            dataset_type: [DatasetConfig(**entry) for entry in entries]
            for dataset_type, entries in data["datasets"].items()
        },
    )


def save_config_to_output_dir(config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as file:
        yaml.dump(
            {
                key: vars(value) if hasattr(value, "__dict__") else value
                for key, value in vars(config).items()
            },
            file,
            default_flow_style=False,
            allow_unicode=True,
        )

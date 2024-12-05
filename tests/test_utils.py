from krx_lm_train.utils import load_config


def test_load_config():
    # Path to your configuration file
    config_path = "configs/sft_config.yaml"
    config = load_config(config_path)

    # Test ModelConfig
    assert config.model.name == "unsloth/Qwen2-0.5B-Instruct-bnb-4bit"
    assert config.model.max_seq_length == 2048
    assert config.model.dtype is None
    assert config.model.load_in_4bit is True

    # Test LoRAConfig
    assert config.lora.r == 64
    assert config.lora.target_modules == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert config.lora.lora_alpha == 16
    assert config.lora.lora_dropout == 0
    assert config.lora.bias == "none"
    assert config.lora.use_gradient_checkpointing == "unsloth"
    assert config.lora.random_state == 42
    assert config.lora.use_rslora is True
    assert config.lora.loftq_config is None

    # Test TrainingConfig
    assert config.training.per_device_train_batch_size == 4
    assert config.training.gradient_accumulation_steps == 4
    assert config.training.warmup_ratio == 0.05
    assert config.training.num_train_epochs == 2
    assert config.training.learning_rate == 0.0002
    assert config.training.logging_steps == 10
    assert config.training.optim == "adamw_8bit"
    assert config.training.weight_decay == 0.01
    assert config.training.lr_scheduler_type == "linear"
    assert config.training.seed == 42
    assert config.training.output_dir == "experiments-001"
    assert config.training.save_steps == 200
    assert config.training.eval_steps == 200
    assert config.training.evaluation_strategy == "steps"
    assert config.training.report_to == "none"

    # Test DatasetConfig
    assert "instruction" in config.datasets
    assert "mcqa" in config.datasets

    instruction_datasets = config.datasets["instruction"]
    assert len(instruction_datasets) == 1
    assert instruction_datasets[0].name == "amphora/krx-sample-instructions"
    assert instruction_datasets[0].ratio == 1

    mcqa_datasets = config.datasets["mcqa"]
    assert len(mcqa_datasets) == 2
    assert mcqa_datasets[0].name == "TwoSubPlace/financial_accounting_mcqa_synthetic2"
    assert mcqa_datasets[0].ratio == 1
    assert mcqa_datasets[1].name == "TwoSubPlace/financial_market_mcqa_synthetic2"
    assert mcqa_datasets[1].ratio == 1

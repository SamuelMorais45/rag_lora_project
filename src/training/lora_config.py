from peft import LoraConfig, TaskType

def make_lora_config(cfg: dict, task_type: TaskType) -> LoraConfig:
    return LoraConfig(
        task_type      = task_type,
        r              = cfg["lora_r"],
        lora_alpha     = cfg["lora_alpha"],
        lora_dropout   = cfg["lora_dropout"],
        target_modules = cfg["target_modules"],
        bias           = "none",
        inference_mode = False,
    )
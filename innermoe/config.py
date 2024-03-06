from peft import LoraConfig
import os
from base_config import *

out_dir = os.path.join(out_dir, 'innermoe')
os.makedirs(out_dir, exist_ok=True)

eval_samples = 500
max_no_adding_times = 20
steps_per_eval = 100

lora_config = LoraConfig(
            r=32, 
            lora_alpha=32, 
            target_modules=["q_proj", "v_proj"], 
            lora_dropout=0.05, 
            bias="none",
        )

phase_lrs = [1e-7, 5e-5, 1e-6]
phase1_data_rate = 0.6
val_rate = 0.3

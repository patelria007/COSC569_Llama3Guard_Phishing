# config.py

# Model and device configurations
# model_id = "NousResearch/Llama-2-7b-hf"
model_id = "meta-llama/Meta-Llama-3-8B"

run = 0
final_model_name = f"Llama3-QLORA-phishing-{run}"
final_model_path = f"models/{final_model_name}"

# Load the entire model on the GPU 0
device_map = {"": 0}

# Divide into batches
# batch_size = 128

################################################################################
# bitsandbytes parameters
################################################################################
# Activate 4-bit precision base model loading
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Activate nested quantization for 4-bit base models (double quantization)
use_double_nested_quant = False

################################################################################
# QLoRA parameters
################################################################################
# LoRA attention dimension
lora_r = 64
# Alpha parameter for LoRA scaling
lora_alpha = 16
# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# TrainingArguments parameters
################################################################################
# Output directory where the model predictions and checkpoints will be stored
output_dir = final_model_path
# Number of training epochs
num_train_epochs =  5 #15
# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = True
bf16 = False
# Batch size per GPU for training
# per_device_train_batch_size = 4
per_device_train_batch_size = 8
# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1
# Enable gradient checkpointing
gradient_checkpointing = True
# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3
# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4
# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001
# Optimizer to use
optim = "paged_adamw_32bit"
# Learning rate schedule
lr_scheduler_type = "cosine"
# Number of training steps (overrides num_train_epochs)
max_steps = -1
# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03
# Group sequences into batches with same length
group_by_length = False
# Save checkpoint every X updates steps
save_steps = 0
# Log every X updates steps
logging_steps = 25
# Disable tqdm
disable_tqdm = False

################################################################################
# SFTTrainer parameters
################################################################################
# Maximum sequence length to use
max_seq_length = 1024
# Pack multiple short examples in the same input sequence to increase efficiency
packing = True

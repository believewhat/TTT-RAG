import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import deepspeed
from transformers import default_data_collator
from torch.optim import AdamW
from transformers import GenerationConfig, TextStreamer
import torch.distributed as dist
import re
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
def setup_distributed():
    """Setup distributed environment for torchrun."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

# 配置DeepSpeed和LoRA
def prepare_model_and_tokenizer(model_name_or_path, lora_config, cache_dir, trainable_params):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_dir, torch_dtype=torch.float16, trust_remote_code=True)

    # 添加特殊符号
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

    # 配置LoRA
    model = get_peft_model(model, lora_config)

    # 冻结不需要训练的参数
    for name, param in model.named_parameters():
        if not any(k in name for k in trainable_params.split(",")):
            param.requires_grad = False

    # 启用 gradient_checkpointing
    model.config.use_cache = False  # 禁用缓存，适配 gradient_checkpointing
    model.gradient_checkpointing_enable()
    

    return model, tokenizer

# 数据集类
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": item["input"]},
            {"role": "assistant", "content": item["output"]}  # 拼一起
        ]
        # 拼接 prompt 和 answer，一起 encode
        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tokenized = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # 找到 answer 的开始位置
        answer_start = full_prompt.rfind(item["output"])
        if answer_start == -1:
            raise ValueError("输出字符串没有出现在拼接的 prompt 中")

        # 找到从 prompt 开始到 answer_start 之间的 token 数目
        prefix = full_prompt[:answer_start]
        prefix_token_count = len(self.tokenizer(prefix)["input_ids"])

        # 构建 labels，前面的都设为 -100（被 mask）
        labels = input_ids.clone()
        labels[:prefix_token_count] = -100  # 忽略非 answer 的 token

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
# 训练函数
def train_model(model, dataset, ds_config_path, num_epochs, batch_size, tokenizer, device, learning_rate):
    model.train()
    train_dataset = CustomDataset(dataset, tokenizer, max_length=2048)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=default_data_collator)

    # 初始化 DeepSpeed
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 初始化 DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,  # 传递优化器
        model_parameters=model.parameters(),
        config=ds_config_path
    )

    # 训练循环
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model_engine(**batch)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()
        print(f"Epoch {epoch + 1}/{num_epochs} completed.")
    del model_engine

# 推理函数
def generate_response(prompt, model, tokenizer, max_length=200, device="cuda"):
    model.eval()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
    streamer = TextStreamer(tokenizer)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.9,
            top_p=0.9,
            use_cache=True,
            streamer=streamer,
            repetition_penalty=1.2
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)
def compute_token_stats(dataset, tokenizer, max_length=2048):
    input_token_count = 0
    output_token_count = 0
    total_samples = 0

    for item in dataset:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": item["input"]},
            {"role": "assistant", "content": item["output"]}
        ]

        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # 分别 tokenize 输入和输出部分
        answer_start = full_prompt.rfind(item["output"])
        if answer_start == -1:
            print(f"Warning: Output not found in prompt for sample {total_samples}")
            continue

        prefix = full_prompt[:answer_start]
        suffix = full_prompt[answer_start:]

        input_tokens = tokenizer(prefix, truncation=True, max_length=max_length)["input_ids"]
        output_tokens = tokenizer(suffix, truncation=True, max_length=max_length)["input_ids"]

        input_token_count += len(input_tokens)
        output_token_count += len(output_tokens)
        total_samples += 1

    print("✅ Token Stats Summary")
    print(f"Total samples: {total_samples}")
    print(f"Total input tokens (prompt): {input_token_count}")
    print(f"Total output tokens (answer): {output_token_count}")
    print(f"Avg input tokens per sample: {input_token_count / total_samples:.2f}")
    print(f"Avg output tokens per sample: {output_token_count / total_samples:.2f}")
# 主函数
def main():
    # 配置路径
    #deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
    model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
    mode = 'qwen_7B'
    cache_dir = "./cache/"
    data_path = "./output/final_dataset_medmcqa_test.json"
    ds_pretrain_config = "/home/jwang/Project/uptodata/medical_rules/train_colbert/ColBERT/ds_configs/stage2.json"
    ds_finetune_config = "/home/jwang/Project/uptodata/medical_rules/train_colbert/ColBERT/ds_configs/stage2.json"
    trainable_params = "q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj,lm_head"
    output_dir = "./output"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    #local_rank = setup_distributed()
    # LoRA配置
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj', 'lm_head'],
        lora_dropout=0.1,
        bias="none"
    )
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    flat_finetune_data = [sample for item in data for sample in item['train_finetune']]
    compute_token_stats(flat_finetune_data, tokenizer)
    # 推理和训练
    answers = []
    model, tokenizer = prepare_model_and_tokenizer(model_name_or_path, lora_config, cache_dir, trainable_params)
    #model.to(device)
    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    initial_state_dict = model.state_dict()
    try:
        with open(f'./log_{mode}/log_medmcqa.json', 'r') as file:
            answers = json.load(file)
    except:
        answers = []
    for i, item in enumerate(data):
        if i < len(answers):
            continue
        model.load_state_dict(initial_state_dict)
        
        # 预训练
        #print("Starting pretraining...")
        #train_model(model, item['train_pretrain'], ds_pretrain_config, num_epochs=1, batch_size=2, tokenizer=tokenizer, device=device, learning_rate=2e-5)
        # 微调
        print("Starting fine-tuning...")
        train_model(model, item['train_finetune'][:512], ds_finetune_config, num_epochs=1, batch_size=2, tokenizer=tokenizer, device=device, learning_rate=2e-5)

        if local_rank == 0:
            # 推理
            #logical_rule = 'Logical Chain:\n'.join(item['doc'][:20])
            input_query = item['input']
            template_chain = json.dumps({
                "Chain": "correct logical chain using '->'"
            })
            correct_rule = {}
            for option in ['A', 'B', 'C', 'D']:
                if f"{option}:" not in input_query:
                    continue
                inference_input = (
                    f"""{input_query}
                    The physician said the correct answer is {option}. The physician must be correct. Now you should generate the logical chain based on physician's answer.
                    The return format must be json and don't generate any pther information: {template_chain}
                    """
                )
                logical_chain = ''
                pattern = r'({"Chain": .*)'
                logical_chain = generate_response(inference_input, model, tokenizer, device=device)
                try:
                    # Split the logical_chain to isolate the part after the answer template
                    logical_chain = logical_chain.split(template_chain)[1]
            
                    # Search for the JSON-like structure
                    match = re.search(pattern, logical_chain, re.DOTALL)

                    if match:
                        json_content = match.group(1)

                        # Check if the closing brace is missing
                        if not json_content.endswith('}'):
                            json_content += '" }'  # Add missing closing brace

                        correct_rule[option] = json.loads(json_content.replace('\n', ''))
                    else:
                        # If no match is found, reset logical_chain and continue
                        correct_rule[option] = logical_chain
                except Exception as e:
                    correct_rule[option] = logical_chain
            answers.append({'id': i, 'Chain': correct_rule})

            with open(f'./log_{mode}/log_medmcqa.json', 'w', encoding='utf-8') as file:
                json.dump(answers, file, ensure_ascii=False, indent=4)
if __name__ == "__main__":
    main()

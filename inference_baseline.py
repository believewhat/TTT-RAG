import os
import json
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams



# 构造 prompt（user only）
def generate_prompt(question):
    messages = [
        {"role": "user", "content": f"You are a medical AI assistant. Answer multiple-choice questions by outputting the logical chain (your inference step, each step is divided by '->'). Please return in English!\n{question}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

# 推理函数
def generate_response(prompt):
    result = model.generate([prompt], sampling_params)[0]
    return result.outputs[0].text.strip()

if __name__ == "__main__":
    # 模型和 tokenizer 设置
    MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
    mode = "14"
    data = "medqa"
    output_path = f"qwen2.5_{mode}B_{data}_logical.json"
    input_path = "/home/jwang/Project/rStar/data/MedQA/test.json"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = LLM(model=MODEL_NAME, max_model_len=2000)

    # 推理参数
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.0,
        max_tokens=100,
        stop=["<|im_end|>"]
    )   
    

    with open(input_path, "r") as f:
        data = json.load(f)

    results = []
    for i, example in enumerate(data):
        question = example["problem"].replace('.,', '')
        prompt = generate_prompt(question)
        try:
            response = generate_response(prompt)
            example["prediction"] = response
            print(f"[{i+1}/{len(data)}] ✓ ID {example['id']} → {response}")
        except Exception as e:
            example["prediction"] = "ERROR"
            print(f"[{i+1}/{len(data)}] ✗ ID {example['id']} 出错: {e}")

        results.append(example)

        if (i + 1) % 50 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ 所有问题处理完毕，结果已保存至 {output_path}")
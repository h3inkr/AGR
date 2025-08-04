import argparse
import json
from tqdm import tqdm
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import string, re

from metrics import compute_hit_at_k

# 🔄 프롬프트 수정
SystemEvaluatePrompt = """\
You are a helpful assistant for multiple-choice question answering. \
You are given a question and 4 options labeled A, B, C, D and E. \
Your task is to answer only with the correct choice label: A, B, C, D or E. \
Do not provide explanations or repeat the question. Just output one of: A, B, C, D or E.\
"""

UserEvaluatePrompt = """
Question: {question}
Choices: 
(A) {answer_choices[0]} 
(B) {answer_choices[1]} 
(C) {answer_choices[2]} 
(D) {answer_choices[3]}
(E) {answer_choices[4]}

Please answer only with the correct choice letter: A, B, C, D or E.
"""

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_argument()

    model_name = "meta-llama/Meta-Llama-3-8B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config
    )

    data = []
    with open(args.eval_path, "r") as f:
        json_data = json.load(f)
        for d in json_data:
            data.append(d)

    mcq = []
    with open(args.data_path, "r") as f:
        json_data = json.load(f)
        for d in json_data:
            mcq.append(d)

    responses = []
    total = len(data)
    hit_cnt_5, hit_cnt_100, correct = 0, 0, 0
    for idx in tqdm(range(len(data)), desc="Evaluating..."):
        #if idx >= 10:
        #    break
        example = data[idx]
        choices = mcq[idx]
        question = example["enriched_ex"]
        answers = choices["answers"]
        all_hits = example["all_hits"]
        answer_choices = choices["choices"]["text"]

        # hit@k
        hit_5 = compute_hit_at_k(all_hits, 5)
        if hit_5:
            hit_cnt_5 += 1

        hit_100 = compute_hit_at_k(all_hits, 100)
        if hit_100:
            hit_cnt_100 += 1

        # acc
        inputs = UserEvaluatePrompt.format(
            question=question,
            answer_choices=answer_choices,
        )

        messages = [
            {
                "role": "system",
                "content": SystemEvaluatePrompt
            },
            {
                "role": "user",
                "content": inputs
            }
        ]

        # prompt 생성
        prompt = f"{SystemEvaluatePrompt}\nUser: {inputs}\nAssistant:"
        inputs_tokenized = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(
            **inputs_tokenized,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=5
        )

        # 🔄 후처리로 A-E 중 하나만 추출
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        response_line = decoded.split("Assistant:")[-1].strip()
        match = re.search(r"\b([A-E])\b", response_line)
        response = match.group(1) if match else "INVALID"
        #print(f"Model Response: {response_line} -> Final: {response}")

        responses.append(response)

        print(f"Model Response: {response}, Ground truth: {answers[0]}")
        if normalize_answer(answers[0]) == normalize_answer(response):
            correct += 1

    hit_at_5 = hit_cnt_5 / total * 100
    hit_at_100 = hit_cnt_100 / total * 100
    acc = correct / total * 100

    print(f"🔑 Hit@5: {round(hit_at_5, 2)}%")
    print(f"🔑 Hit@100: {round(hit_at_100, 2)}%")
    print(f"🔑 Accuracy: {round(acc, 2)}%")

import torch
from vllm import LLM, SamplingParams
import random
import os
import argparse
import json
from tqdm import tqdm
from collections import defaultdict

def expand_enriched(query: str, cont_ref: str, llm: LLM) -> list:
    prompt = f"""[INST]Question: {query}
Contextual references: {cont_ref}

Based on the contextual references and your available knowledge, create a possibly correct and concise answer that directly answers the question "{query}".

Expected Output:
"Answer": answer with a detailed context
Output:
"Answer":[/INST]"""

    sampling_params = SamplingParams(
        temperature=0.8,
        repetition_penalty=1.1,
        max_tokens=100,
        n=10
    )

    output = llm.generate(prompts=prompt, sampling_params=sampling_params)
    expanded = []
    expanded += [output[0].outputs[i].text.replace('\n', ' ').strip().strip('Answer: ') for i in range(10)]
    random.shuffle(expanded)

    return expanded

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieved_path", "-rp", type=str, required=True)
    parser.add_argument("--output_path", "-op", type=str, required=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_argument()
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm = LLM(model=model_name, tensor_parallel_size=1, max_model_len=8192)

    with open(args.retrieved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    grouped_data = defaultdict(lambda: {"answers": None, "ctxs": []})
    for example in data:
        q = example["original_q"]
        grouped_data[q]["answers"] = example["answers"]
        grouped_data[q]["ctxs"].extend(example["ctxs"])
    #print(grouped_data)

    for query, group in tqdm(grouped_data.items(), desc="Enriched Expanding..."):
        answers = group["answers"]
        #ref = " ".join([ctx['text'] for ctx in group["ctxs"]])
        #print(ref)
        
        # context 섞기
        shuffled_ctxs = group['ctxs'].copy()
        random.shuffle(shuffled_ctxs)

        # 상위 10개 문서 선택 및 포맷팅
        ref = "\n".join(
            [f"Reference context: {ctx['text']}" for ctx in shuffled_ctxs[:10]]
        )

        enriched = expand_enriched(query, ref, llm)
        #print(enriched)

        if args.output_path:
            with open(args.output_path, "a") as save_file:
                saved = {
                    "original_question": query,
                    "answers": answers,
                    "second_ex": enriched
                }
                json.dump(saved, save_file)
                save_file.write("\n")


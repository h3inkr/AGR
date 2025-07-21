import torch
import os
import re
import random
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
import argparse

def refine(query: str, candidates: str, llm: LLM) -> str:
    prompt = f"""[INST]Question: {query}
Candidate answer list:
{candidates}

Based on the candidate answers, please evaluate the accuracy and reliability of each candidate answer. Identify any misinformation or incorrect facts in the answers. Please use all your available knowledge to verify the accuracy of these candidate answers. Then, generate a correct and concise response that best answer the question, refer to the information from the candidate answers that you have verified as accurate.

Expected Output:
"Best Answer": a concise answer for the question "{query}"
"Explanation": 
Output:
Best Answer: [/INST]"""

    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=300,
        repetition_penalty=1.1
    )

    output = llm.generate(prompts=prompt, sampling_params=sampling_params)
    refined = output[0].outputs[0].text.replace('\n', ' ').strip().strip('Best Answer: ')

    match = re.search(r"(.*?)\s*Explanation:", refined, re.IGNORECASE)
    best_result = match.group(1) if match else None
    if best_result:
        refined_detail = f"Match: {best_result}"
    else:
        best_result = candidates[random.randint(0, 9)]
        refined_detail = f"Not match, candidate: {best_result}"

    return refined_detail

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_path", "-cp", type=str, required=True)
    parser.add_argument("--output_path", "-op", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_argument()
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm = LLM(model=model_name, tensor_parallel_size=1, max_model_len=8192)
    
    data = []
    with open(args.candidate_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    for idx in tqdm(range(len(data)), desc="Refining..."):
        example = data[idx]
        query = example["original_question"]
        answers = example["answers"]
        candidates = "\n".join(
            [f"Candidate Answer {str(i+1)}: {temp.strip()}" for i, temp in enumerate(example["second_ex"])]
        )

        refined = refine(query, candidates, llm)

        if args.output_path:
            with open(args.output_path, "a") as save_file:
                saved = {
                    "original question": query,
                    "answers": answers,
                    "candidates": example["second_ex"],
                    "refined": refined
                }
                json.dump(saved, save_file)
                save_file.write("\n")
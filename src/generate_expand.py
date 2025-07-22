import torch
from vllm import LLM, SamplingParams
import random
import os
import argparse
import json
from tqdm import tqdm

def expand_question(query: str, answer_analysis: str, llm: LLM) -> list:
    prompt = f"""[INST]Question: {query}
Question analysis: {answer_analysis}

Based on the analysis and your available knowledge, create a possibly correct and concise answer that directly answers the question "{query}".

Expected Output:
"Answer": answer with a detailed context
Output:
"Answer":[/INST]"""

    sampling_params = SamplingParams(
        temperature=0.8,
        repetition_penalty=1.1,
        max_tokens=100,
        n=30
    )

    output = llm.generate(prompts=prompt, sampling_params=sampling_params)
    expanded = []
    expanded += [output[0].outputs[i].text.replace('\n', ' ').strip().strip('Answer: ') for i in range(15)]
    random.shuffle(expanded)

    return expanded

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyzed_path", "-dp", type=str, required=True)
    parser.add_argument("--output_path", "-op", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_argument()
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    llm = LLM(model=model_name, tensor_parallel_size=1, max_model_len=4096)

    data = []
    with open(args.analyzed_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    for idx in tqdm(range(len(data)), desc="Expanding..."):
        example = data[idx]
        query = example["question"]
        answers = example["answers"]
        analysis = example["analysis"]

        expanded = expand_question(query, analysis, llm)

        if args.output_path:
            with open(args.output_path, "a") as save_file:
                for ex in expanded:
                    saved = {
                        "original_question": query,
                        "answers": answers,
                        "first_ex": ex,
                    }
                    json.dump(saved, save_file)
                    save_file.write("\n")
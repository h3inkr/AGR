import torch
from vllm import LLM, SamplingParams
import argparse
import json
from tqdm import tqdm

def generate_docs(query: str, llm: LLM) -> str:
    prompt =f"""[INST] Write a passage that answers the given query:
    
Question: {query}

Passage:[/INST]"""

    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=300,
        repetition_penalty=1.1
    )

    output = llm.generate(prompts=prompt, sampling_params=sampling_params)
    expanded = output[0].outputs[0].text.replace('\n', ' ').strip()

    return expanded

def expand_questions(query: str, pseudo_docs: str, n: int) -> str:
    q_repeated = " ".join([query] * n)
    
    return q_repeated + " " + pseudo_docs

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str, required=True)
    parser.add_argument("--output_path", "-op", type=str)
    parser.add_argument("--n", "-n", type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_argument()
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm = LLM(model=model_name, tensor_parallel_size=1, max_model_len=4096)

    data = []
    with open(args.data_path, 'r') as f:
        json_data = json.load(f)
        for d in json_data:
            data.append(d)

    for idx in tqdm(range(len(data)), desc="Expanding..."):
        example = data[idx]
        query = example["question"]
        answers = example["answers"]

        pseudo_docs = generate_docs(query, llm)
        expanded = expand_questions(query, pseudo_docs, args.n)

        saved = {
            "question": query,
            "answers": answers,
            "expanded": expanded
        }

        if args.output_path:
            with open(args.output_path, "a") as save_file:
                json.dump(saved, save_file)
                save_file.write("\n")

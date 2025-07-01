import torch
from vllm import LLM, SamplingParams
import random
import argparse
import json
from tqdm import tqdm

def expansion_first(query: str, answer_analysis: str, llm: LLM) -> str:

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
        n=15
    )

    output = llm.generate(prompts=prompt, sampling_params=sampling_params)
    expanded = []
    expanded += [output[0].outputs[i].text.replace('\n', ' ').strip().strip('Answer: ') for i in range(15)]
    random.shuffle(expanded)
    print(expanded)
    #output_text = output[0].outputs[0].text
    #output_text = output_text.replace('\n', ' ').strip()

    return expanded
'''
def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str, required=True)
    parser.add_argument("--output_path", "-op", type=str)
    args = parser.parse_args()
    return args
'''
if __name__ == "__main__":
   # args = parse_argument()
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    llm = LLM(model=model_name, tensor_parallel_size=1, max_model_len=4096)
    expanded = expansion_first("what does hp mean in war and order", "The question is asking for the meaning of the abbreviation 'HP' in relation to the game 'War and Order'.", llm)
    print(expanded)

'''
    data = []
    with open(args.data_path, 'r') as f:
        json_data = json.load(f)
        for d in json_data:
            data.append(d)

    for idx in tqdm(range(len(data)), desc="Analyzing..."):
        example = data[idx]
        query = example["question"]
        analysis = example["analysis"]

        expanded = expansion_first(query, analysis, llm)

        saved = {
            "question": query,
            
        }

        if args.output_path:
            with open(args.output_path, "a") as save_file:
                json.dump(saved, save_file)
                save_file.write("\n")
'''
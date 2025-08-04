import torch
from vllm import LLM, SamplingParams
import argparse
import json
from tqdm import tqdm

def extract_keywords(query: str, llm: LLM) -> str:

    prompt = f"""[INST]Please note that this is a brand new conversation start. When responding to the following questions, disregard all previous interactions and context.
Question: {query}

When analyzing a phrase, first consider if the phrase could be a proper noun, such as the title of a song, movie, book, or other work. If it is a common phrase or doesn't immediately appear to be a title, then proceed to analyze its grammatical structure as a standard phrase. However, if there is a possibility that the phrase is a title, treat it as a proper noun and analyze it in that context.

Do not attempt to explain or answer the question, just provide the key phrases.

Expected Output:
"Key Phrases Output": key phrases in "{query}"

Output:[/INST]"""

    sampling_params = SamplingParams(
        temperature=0.2,
        repetition_penalty=1.1,
        max_tokens=150
    )

    output = llm.generate(prompts=prompt, sampling_params=sampling_params)
    output_text = output[0].outputs[0].text
    output_text = output_text.replace('\n', ' ').strip()

    return output_text

def analyze(query: str, answer_kp_analysis: str, llm: LLM) -> str:
    prompt = f"""[INST]Question: {query}
Key Phrases in query:{answer_kp_analysis}

Analyze the question carefully. Determine what type of information is being asked for. Consider the most direct way to find this information. If the question is about identifying something or someone, focus on the specific details provided. Avoid assumptions or interpretations beyond what is explicitly asked. Provide a clear and concise answer based on the analysis.

Do not attempt to explain or answer the question, just provide the Question Analysis.

Expected Output:
"Question Analysis": Question Analysis based on Key Phrases

Output:[/INST]"""

    sampling_params = SamplingParams(
        temperature=0.2,
        repetition_penalty=1.1,
        max_tokens=150
    )
    output = llm.generate(prompts=prompt, sampling_params=sampling_params)
    output_text = output[0].outputs[0].text
    output_text = output_text.replace('\n', ' ').strip()

    return output_text
    
def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str, required=True)
    parser.add_argument("--output_path", "-op", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_argument()
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    llm = LLM(model=model_name, tensor_parallel_size=1, max_model_len=4096)

    data = []
    with open(args.data_path, 'r') as f:
        json_data = json.load(f)
        for d in json_data:
            data.append(d)

    for idx in tqdm(range(len(data)), desc="Analyzing..."):
        example = data[idx]
        query = example["question"]
        answers = example["answers"]

        keywords = extract_keywords(query, llm)
        analysis = analyze(query, keywords, llm)

        saved = {
            "question": query,
            "answers": answers,
            "analysis": analysis
        }

        if args.output_path:
            with open(args.output_path, "a") as save_file:
                json.dump(saved, save_file)
                save_file.write("\n")
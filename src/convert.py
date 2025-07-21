import jsonlines
from tqdm import tqdm
import argparse
import json

def convert_arc(data) -> dict:
    return {
        "question": data["question"],
        "answers": [data["answerKey"]],
        "choices": {
            "text": data["choices"]["text"],
            "label": data["choices"]["label"]
        },
        "qa_pairs": "null",
        "ctxs": []
    }

def convert_csqa(data) -> dict:
    return {
        "question": data["question"]["stem"],
        "answers": [data["answerKey"]],
        "choices": {
            "text": [choice["text"] for choice in data["question"]["choices"]],
            "label": [choice["label"] for choice in data["question"]["choices"]]
        },
        "qa_pairs": "null",
        "ctxs": []
    }

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-ip", type=str, required=True)
    parser.add_argument("--output_path", "-op", type=str, required=True)
    parser.add_argument("--dataset_name", "-d", type=str, required=True, choices=["arc", "csqa", "2wikimh"])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_argument()

    data = []
    with jsonlines.open(args.input_path) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            data.append(d)

    converted = []
    for idx in tqdm(range(len(data)), desc="Converting..."):
        example = data[idx]
        if args.dataset_name == "arc":
            converted.append(convert_arc(example))
        elif args.dataset_name == "csqa":
            converted.append(convert_csqa(example))
        
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

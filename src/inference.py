import argparse
import json
from tqdm import tqdm

from metrics import compute_hit_at_k

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_argument()

    data = []
    with open(args.eval_path, "r") as f:
        json_data = json.load(f)
        for d in json_data:
            data.append(d)

    total = len(data)
    hit_cnt_5, hit_cnt_100 = 0, 0
    for idx in tqdm(range(len(data)), desc="Evaluating..."):
        example = data[idx]
        answers = example["answers"]
        all_hits = example["all_hits"]
        retrieved = example["ctxs"]
        
        # hit@k
        hit_5 = compute_hit_at_k(all_hits, 5)
        if hit_5:
            hit_cnt_5 += 1

        hit_100 = compute_hit_at_k(all_hits, 100)
        if hit_100:
            hit_cnt_100 += 1

        # em@k

    hit_at_5 = hit_cnt_5 / total * 100
    hit_at_100 = hit_cnt_100 / total * 100

    print(f"🔑 Hit@5: {round(hit_at_5, 2)}%")
    print(f"🔑 Hit@100: {round(hit_at_100, 2)}%")

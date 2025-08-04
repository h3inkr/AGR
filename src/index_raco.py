import os
import json
import argparse
from pyserini.search.lucene import LuceneSearcher
import subprocess

def convert_tsv_to_jsonl(tsv_path, jsonl_path):
    with open(tsv_path, "r", encoding="utf-8") as f_in, open(jsonl_path, "w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if i == 0:  # 첫 줄 헤더는 건너뛰기
                continue
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            docid, text = parts[0], parts[1]
            json.dump({"id": docid, "contents": text}, f_out, ensure_ascii=False)
            f_out.write("\n")
    print(f"✅ Converted {tsv_path} → {jsonl_path}")

def build_index(jsonl_path, index_dir):
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    input_dir = os.path.dirname(jsonl_path) or "."
    cmd = [
        "python", "-m", "pyserini.index",
        "--collection", "JsonCollection",
        "--input", input_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "8",
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]
    print("✅ Building index...")
    subprocess.run(cmd, check=True)
    print(f"✅ Index built at {index_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_path", type=str, required=True, help="Input raco.tsv path")
    parser.add_argument("--output_jsonl", type=str, default="raco.jsonl", help="Output JSONL path")
    parser.add_argument("--index_dir", type=str, default="raco_index", help="Index output directory")
    args = parser.parse_args()

    #convert_tsv_to_jsonl(args.tsv_path, args.output_jsonl)
    build_index(args.output_jsonl, args.index_dir)

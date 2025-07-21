import torch
import json
from pathlib import Path
import glob
import functools
from vllm import LLM, SamplingParams
import random
import os
import argparse
from tqdm import tqdm
from retriever_utils import load_passages, SparseRetriever, validate, save_results

def parse_qa_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            yield item["original_question"], item["answers"], item['first_ex']
            
def get_datasets(qa_file_pattern):
    all_patterns = qa_file_pattern.split(",")
    all_qa_files = functools.reduce(lambda a, b: a + b, [glob.glob(p) for p in all_patterns])

    qa_file_dict = {}
    for qa_file in all_qa_files:
        dataset_name = Path(qa_file).stem.split("_")[-1]
        dataset = list(parse_qa_file(qa_file))

        questions, question_answers, expanded = [], [], []
        for question, answers, first_ex in dataset:
            questions.append(question)          
            question_answers.append(answers)
            expanded.append(first_ex)

        qa_file_dict[dataset_name] = (questions, question_answers, expanded)

    return qa_file_dict

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded_path", "-dp", type=str, required=True)
    parser.add_argument("--passage", "-pa", type=str, required=True)
    parser.add_argument("--index_name", type=str, default="wikipedia-dpr")
    parser.add_argument("--use_rm3", action="store_true")
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--dedup", action="store_true")
    parser.add_argument("--result_file_path", type=str, required=True)
    parser.add_argument("--n_top_docs", type=int, default=3)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_argument()

    qa_dict = get_datasets(args.expanded_path)
    passages = load_passages(args.passage)
    retriever = SparseRetriever(args.index_name, args.use_rm3, args.num_threads, args.dedup)

    for idx, (dataset_name, (questions, question_answers, expanded)) in enumerate(tqdm(qa_dict.items())):
        #if idx >= 3:
        #    break

        top_ids_and_scores = retriever.get_top_docs(questions=expanded, top_docs=args.n_top_docs)
        questions_doc_hits = validate(
            dataset_name,
            passages,
            question_answers,
            top_ids_and_scores,
            args.num_threads,
            "string",
        )

        save_results(
            passages,
            questions,
            expanded,
            question_answers,
            top_ids_and_scores,
            questions_doc_hits,
            args.result_file_path,
        )
import torch
import json
from pathlib import Path
import glob
import functools
from tqdm import tqdm
import argparse
from retriever_utils import load_passages, SparseRetriever, validate, save_results_eval

def parse_qa_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            yield item["original question"], item["refined"]

def parse_original_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)  # 이미 dict 또는 list[dict]

        for idx, item in enumerate(json_data):
            answer_label = item["answers"][0]  # 예: "C"
            labels = item["choices"]["label"]
            texts = item["choices"]["text"]

            if answer_label in labels:
                index = labels.index(answer_label)
                correct_answer_text = texts[index]
                yield correct_answer_text

def get_datasets(qa_file_pattern, dataset_name, original_path):
    all_patterns = qa_file_pattern.split(",")
    all_qa_files = functools.reduce(lambda a, b: a + b, [glob.glob(p) for p in all_patterns])

    qa_file_dict = {}
    for qa_file in all_qa_files: 
        dataset = list(parse_qa_file(qa_file))
        original = list(parse_original_file(original_path))
        #print(original)

        questions, refined, question_answers = [], [], []
        for (question, refined_q), answers in zip(dataset, original):
            questions.append(question)
            refined.append(refined_q)
            question_answers.append(answers)

        qa_file_dict[dataset_name] = (questions, refined, question_answers)
    
    return qa_file_dict

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True, help="Original file path")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--passage", "-pa", type=str, required=True)
    parser.add_argument("--index_name", type=str, default="wikipedia-dpr")
    parser.add_argument("--use_rm3", action="store_true")
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--dedup", action="store_true")
    parser.add_argument("--result_file_path", type=str, required=True)
    parser.add_argument("--n_top_docs", type=int, default=100)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_argument()
    ### retrieval 100개씩

    qa_dict = get_datasets(args.qa_path, args.dataset_name, args.data_path)
    passages = load_passages(args.passage)
    retriever = SparseRetriever(args.index_name, args.use_rm3, args.num_threads, args.dedup)

    for idx, (dataset_name, (questions, refined, question_answers)) in enumerate(tqdm(qa_dict.items())):
        top_ids_and_scores = retriever.get_top_docs(questions=refined, top_docs=args.n_top_docs)
        question_doc_hits = validate(
            dataset_name,
            passages,
            question_answers,
            top_ids_and_scores,
            args.num_threads,
            "string",
        )

        save_results_eval(
            passages,
            questions,
            refined,
            question_answers,
            top_ids_and_scores,
            question_doc_hits,
            args.result_file_path,
        )
import torch
import json
from pathlib import Path
import glob
import functools
from tqdm import tqdm
import argparse
import sys
sys.path.append("src")
from retriever_utils import load_passages, SparseRetriever, validate, save_results_base

def parse_qa_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 전체 JSON 배열을 로드
        for item in data:
            yield item["question"], item["answers"]

def parse_qa_file_mcq(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # JSON 배열 로드
        for item in data:
            question = item["question"]
            
            # label과 text 매핑 생성
            label_to_text = dict(zip(item["choices"]["label"], item["choices"]["text"]))
            
            # 정답 레이블 리스트
            answer_labels = item["answers"]
            
            # 각 정답 레이블을 실제 텍스트로 변환
            answer_texts = [label_to_text[label] for label in answer_labels]
            #print(answer_texts)

            yield question, answer_texts

def get_datasets(qa_file_pattern, dataset_name):
    all_patterns = qa_file_pattern.split(",")
    all_qa_files = functools.reduce(lambda a, b: a + b, [glob.glob(p) for p in all_patterns])

    qa_file_dict = {}
    for qa_file in all_qa_files: 
        dataset = list(parse_qa_file_mcq(qa_file))
        #if dataset_name == 'nq' or '2wiki':
        #    dataset = list(parse_qa_file(qa_file))
        #elif dataset_name == 'csqa':
        #    dataset = list(parse_qa_file_mcq(qa_file))

        questions, question_answers = [], []
        for question, answers in dataset:
            questions.append(question)
            question_answers.append(answers)

        qa_file_dict[dataset_name] = (questions, question_answers)
    
    return qa_file_dict

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_path", type=str, required=True)
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
    #print(args.dataset_name)

    qa_dict = get_datasets(args.qa_path, args.dataset_name)
    passages = load_passages(args.passage)
    retriever = SparseRetriever(args.index_name, args.use_rm3, args.num_threads, args.dedup)

    for idx, (dataset_name, (questions, question_answers)) in enumerate(tqdm(qa_dict.items())):
        top_ids_and_scores = retriever.get_top_docs(questions=questions, top_docs=args.n_top_docs)
        question_doc_hits = validate(
            dataset_name,
            passages,
            question_answers,
            top_ids_and_scores,
            args.num_threads,
            "string",
        )

        save_results_base(
            passages,
            questions,
            question_answers,
            top_ids_and_scores,
            question_doc_hits,
            args.result_file_path,
        )
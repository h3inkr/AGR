from typing import List, Dict, Tuple, Union
import json
from pathlib import Path
import time
import glob
import functools
import csv
from pyserini.search import get_qrels_file, get_topics
from pyserini.search.lucene import LuceneSearcher

import sys
sys.path.append('dpr')
from data.qa_validation import calculate_matches


def parse_qa_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            yield item["question"], item["answers"]

def get_datasets(qa_file_pattern):
    all_patterns = qa_file_pattern.split(",")
    all_qa_files = functools.reduce(lambda a, b: a + b, [glob.glob(p) for p in all_patterns])

    qa_file_dict = {}
    for qa_file in all_qa_files:
        dataset_name = Path(qa_file).stem.split("_")[-1]
        dataset = list(parse_qa_file(qa_file))

        questions, question_answers = [], []
        for first_ex, answers in dataset:
            questions.append(first_ex)
            question_answers.append(answers)

        qa_file_dict[dataset_name] = (questions, question_answers)

    return qa_file_dict

def load_passages(ctx_file: str) -> Dict[object, Tuple[str, str]]:
    docs = {}
    with open(ctx_file) as tsvfile:
        reader = csv.reader(
            tsvfile,
            delimiter="\t",
        )
        # file format: doc_id, doc_text, title
        for row in reader:
            if row[0] != "id":
                docs[row[0]] = (row[1], row[2])
    return docs

class SparseRetriever(object): # BM25
    """
    Does passage retrieving over the provided index and question encoder
    """
    def __init__(
        self,
        index_name,
        use_rm3,
        num_threads,
        dedup=False
    ):  
        self.searcher = LuceneSearcher.from_prebuilt_index(index_name)
        # self.searcher = LuceneSearcher(f'{index_name}lucene9-index.cacm')
        self.use_rm3 = use_rm3
        ##### 启用rm3 QE方法
        if self.use_rm3:
            logger.info(f"Use rm3 QE.") 
            self.searcher.set_rm3()
        self.num_threads = num_threads
        self.dedup = dedup

    def get_top_docs(
        self, questions: List[str], top_docs: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        qids = [str(x) for x in range(len(questions))]
        if self.dedup:
            dedup_q = {}
            dedup_map = {}
            for qid, question in zip(qids, questions):
                if question not in dedup_q:
                    dedup_q[question] = qid
                else:
                    dedup_map[qid] = dedup_q[question]
            dedup_questions = []
            dedup_qids = []
            for question in dedup_q:
                qid = dedup_q[question]
                dedup_questions.append(question)
                dedup_qids.append(qid)
            hits = self.searcher.batch_search(queries=dedup_questions, qids=dedup_qids, k=top_docs, threads=self.num_threads)
            for qid in dedup_map:
                hits[qid] = hits[dedup_map[qid]]
        else:
            hits = self.searcher.batch_search(queries=questions, qids=qids, k=top_docs, threads=self.num_threads)
        time1 = time.time()
        results = []
        for qid in qids:
            example_hits = hits[qid]
            example_top_docs = [hit.docid for hit in example_hits]
            example_scores = [hit.score for hit in example_hits]
            results.append((example_top_docs, example_scores))
        return results

def validate(
    dataset_name: str,
    passages: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
    out_file: str,
    use_wandb: bool = False,
    output_recall_at_k: bool = False,
    log: bool = True
) -> Union[List[List[bool]], Tuple[object, List[float]]]:
    match_stats = calculate_matches(
        passages, answers, result_ctx_ids, workers_num, match_type, log=log
    )

    top_k_hits = match_stats.top_k_hits
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]

    with open(out_file, "w") as f:
        for k, recall in enumerate(top_k_hits):
            f.write(f"{k+1},{recall}\n")

    return match_stats.questions_doc_hits if not output_recall_at_k else (match_stats.questions_doc_hits, top_k_hits)
'''
def save_results(
    passages: Dict[object, Tuple[str, str]],
    original_question: str,
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[List[Tuple[List[object], List[float]]]],
    per_question_hits: List[List[List[bool]]],
    out_file: str,
    output_no_text: bool = False,
):
    
    merged_data = []
    assert len(per_question_hits) == len(questions)
    
    for i, q in enumerate(questions):
        q_answers = answers[i]
        
        for idx in range(len(top_passages_and_scores[i])):
            #results_and_scores = top_passages_and_scores[i]
            results_and_scores = top_passages_and_scores[i][idx]
            #hits = per_question_hits[i][idx]
            hits = per_question_hits[i]
            docs = [passages[doc_id] for doc_id in results_and_scores[0]]
            scores = [str(score) for score in results_and_scores[1]]
            hit_indices = [j+1 for j, is_hit in enumerate(hits) if is_hit]
            hit_min_rank = hit_indices[0] if len(hit_indices) > 0 else None
            #ctxs_num = len(hits[idx])
            ctxs_num = len(hits)

            print(hits)
            print(original_question)
            print(q)
            print(q_answers)
            print(hit_min_rank)
            print(hit_indices)
            print(results_and_scores)
            print(ctxs_num)
            d = {
                "original_question": original_question,
                "expanded_question": q,
                "answers": q_answers,
                "hit_min_rank": hit_min_rank,
                "all_hits": hit_indices,
                "ctxs": [
                    {
                        "id": results_and_scores[0][c],
                        "rank": (c + 1),
                        "title": docs[c][1],
                        "text": docs[c][0] if not output_no_text else "",
                        "score": scores[c],
                        "has_answer": hits[0][c],
                    }
                    for c in range(ctxs_num)
                ],
            }
            merged_data.append(d)
        
    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")

'''
def save_results(
    passages: Dict[object, Tuple[str, str]],
    original_questions: List[str],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
    output_no_text: bool = False,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        hit_indices = [j+1 for j, is_hit in enumerate(hits) if is_hit]
        hit_min_rank = hit_indices[0] if len(hit_indices) > 0 else None
        ctxs_num = len(hits)

        d = {   
                "original_q": original_questions[i],
                "first_ex": q,
                "answers": q_answers,
                "hit_min_rank": hit_min_rank,
                "all_hits": hit_indices,
                "ctxs": [
                    {
                        "id": results_and_scores[0][c],
                        "rank": (c + 1),
                        "title": docs[c][1],
                        "text": docs[c][0] if not output_no_text else "",
                        "score": scores[c],
                        "has_answer": hits[c],
                    }
                    for c in range(ctxs_num)
                ],
            }
        merged_data.append(d)

    if out_file:
        with open(out_file, "w") as writer:
            writer.write(json.dumps(merged_data, indent=4) + "\n")
'''
def save_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
    output_no_text: bool = False,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        hit_indices = [j+1 for j, is_hit in enumerate(hits) if is_hit]
        hit_min_rank = hit_indices[0] if len(hit_indices) > 0 else None
        ctxs_num = len(hits)

        d = {
                "question": q,
                "answers": q_answers,
                "hit_min_rank": hit_min_rank,
                "all_hits": hit_indices,
                "ctxs": [
                    {
                        "id": results_and_scores[0][c],
                        "rank": (c + 1),
                        "title": docs[c][1],
                        "text": docs[c][0] if not output_no_text else "",
                        "score": scores[c],
                        "has_answer": hits[c],
                    }
                    for c in range(ctxs_num)
                ],
            }
        merged_data.append(d)

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
'''

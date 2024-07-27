
"""
Official script for evaluating models built for the Qasper dataset. The script
outputs Answer F1 and Evidence F1 reported in the paper.
"""

import argparse
import json
import re
import string
from collections import Counter

# BertScore and Sentence-Transformer
import torch
import evaluate
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch import nn

bertscore = evaluate.load("bertscore")

def get_answers_and_evidence(data, text_evidence_only):
    answers_and_evidence = {}
    for paper_data in data.values():
        for qa_info in paper_data["qas"]:
            question_id = qa_info["question_id"]
            references = []
            for annotation_info in qa_info["answers"]:
                answer_info = annotation_info["answer"]
                if answer_info["unanswerable"]:
                    references.append(
                        {"answer": "Unanswerable", "evidence": [], "type": "none"}
                    )
                else:
                    if answer_info["extractive_spans"]:
                        answer = ", ".join(answer_info["extractive_spans"])
                        answer_type = "extractive"
                    elif answer_info["free_form_answer"]:
                        answer = answer_info["free_form_answer"]
                        answer_type = "abstractive"
                    elif answer_info["yes_no"]:
                        answer = "Yes"
                        answer_type = "boolean"
                    elif answer_info["yes_no"] is not None:
                        answer = "No"
                        answer_type = "boolean"
                    else:
                        raise RuntimeError(
                            f"Annotation {answer_info['annotation_id']} does not contain an answer"
                        )
                    if text_evidence_only:
                        evidence = [
                            text
                            for text in answer_info["evidence"]
                            if "FLOAT SELECTED" not in text
                        ]
                    else:
                        evidence = answer_info["evidence"]
                    references.append(
                        {"answer": answer, "evidence": evidence, "type": answer_type}
                    )
            answers_and_evidence[question_id] = references

    return answers_and_evidence

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def compute_cosine(prediction, ground_truth, cosine_type, device = 'cpu'):

  tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
  sentence_embedding = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

  encoded_prediction = tokenizer(prediction, padding=True, truncation=True, max_length = 256, return_tensors='pt')
  encoded_groundtruth = tokenizer(ground_truth, padding=True, truncation=True, max_length = 256, return_tensors='pt')

  sentence_embedding.eval()
  sentence_embedding.to(device)
  with torch.no_grad():
    embedded_prediction = sentence_embedding(input_ids = encoded_prediction['input_ids'].to(device),
                                            attention_mask = encoded_prediction['attention_mask'].to(device))

    embedded_groundtruth = sentence_embedding(input_ids = encoded_groundtruth['input_ids'].to(device),
                                            attention_mask = encoded_groundtruth['attention_mask'].to(device))

  pooled_prediction = mean_pooling(embedded_prediction, encoded_prediction['attention_mask'].to(device))
  pooled_groundtruth = mean_pooling(embedded_groundtruth, encoded_groundtruth['attention_mask'].to(device))

  cos = nn.CosineSimilarity(dim=1, eps=1e-6)
  cosine_similarity = cos(pooled_prediction, pooled_groundtruth)

  return cosine_similarity.mean()

def compute_metrics(prediction, ground_truth, bert_type, cosine_type, device = 'cpu'):

  if isinstance(prediction, list) and isinstance(ground_truth[0], list): # Use for Evidence
    prediction = " ".join(prediction)
    sample = []

    for each in ground_truth:
      try : sample.append(each[0])
      except IndexError: continue

    ground_truth = sample

  if len(ground_truth) == 0 : return 0, 0 # The question is unanswerable and the prediction is empty.

  bert_score = bertscore.compute(predictions=[prediction],
                                 references=[ground_truth],
                                 model_type = bert_type,
                                 device = device)

  cosine_similartiy = compute_cosine(prediction = prediction,
                                     ground_truth = ground_truth,
                                     cosine_type = cosine_type,
                                     device = device)

  return np.array([bert_score['f1'][0], cosine_similartiy.item()])


def evaluate(gold, predicted, 
             bert_type = 'bert-base-uncased', 
             cosine_type = 'sentence-transformers/all-MiniLM-L6-v2', 
             retrieval_only=False):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f'Use {device} ')
  answer_score, evidence_score = 0, 0

  for i, (question_id, predicted_item) in enumerate(tqdm(predicted.items())):
    referenced_item = gold[question_id]
    referenced_answers = [each['answer'] for each in referenced_item]
    referenced_evidences = [each['evidence'] for each in referenced_item]

    print('============')
    # Get Bert-score and Cosine Similarity score
    answer_score += compute_metrics(prediction = predicted_item['answer'],
                                  ground_truth = referenced_answers,
                                  bert_type = bert_type,
                                  cosine_type = cosine_type,
                                  device = device)
    evidence_score += compute_metrics(prediction = predicted_item['evidence'],
                                    ground_truth = referenced_evidences,
                                    bert_type = bert_type,
                                    cosine_type = cosine_type,
                                    device = device)

  answer_score = answer_score/len(predicted)
  evidence_score = evidence_score/len(predicted)

  results = {
      'Answer' : {
          'Bert_score' : answer_score[0],
          'Cosine_similarity' : answer_score[1]
      },

      'Evidence' : {
          'Bert_score' : evidence_score[0],
          'Cosine_similarity' : evidence_score[1]
      }
  }

  return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="""JSON lines file with each line in format:
                {'question_id': str, 'predicted_answer': str, 'predicted_evidence': List[str]}""",
    )
    parser.add_argument(
        "--gold",
        type=str,
        required=True,
        help="Test or dev set from the released dataset",
    )

    parser.add_argument(
        "--bert_type",
        help="Select model type for calculating bert-score",
        default="bert-base-uncased",
    )

    parser.add_argument(
        "--cosine_type",
        help="Select model type for calculating cosine similarity",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    parser.add_argument(
        "--retrieval_only",
        help="If set, the evaluator will just evaluate the retrieval scores",
        action="store_true",
    )

    parser.add_argument(
        "--text_evidence_only",
        action="store_true",
        help="If set, the evaluator will ignore evidence in figures and tables while reporting evidence f1",
    )

    args = parser.parse_args()
    gold_data = json.load(open(args.gold))
    gold_answers_and_evidence = get_answers_and_evidence(
        gold_data, args.text_evidence_only
    )
    predicted_answers_and_evidence = {}
    for line in open(args.predictions):
        prediction_data = json.loads(line)
        predicted_answers_and_evidence[prediction_data["question_id"]] = {
            "answer": prediction_data["predicted_answer"],
            "evidence": prediction_data["predicted_evidence"],
        }

    evaluation_output = evaluate(
        gold = gold_answers_and_evidence,
        predicted = predicted_answers_and_evidence,
        bert_type = args.bert_type,
        cosine_type = args.cosine_type,
        retrieval_only=args.retrieval_only,
    )
    print(json.dumps(evaluation_output, indent=2))

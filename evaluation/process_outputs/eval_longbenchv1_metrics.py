import os, json
import re
import string
import argparse
from collections import Counter

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    print(normalized_prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

def scorer(predictions, answers):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        for ground_truth in ground_truths:
            score = max(score, qa_f1_score(prediction, ground_truth))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default='hotpotqa',choices=['hotpotqa', 'musique'])
    parser.add_argument("--model_name", "-m", type=str, default="llama3.1-8b-instruct", choices=['llama3.1-8b-instruct','qwen3-8b'])
    args = parser.parse_args()

    files = os.listdir(f'outputs/{args.model_name}/{args.dataset}/')
    output = ["Partial\t\tScore"]
    scores = []
    for file in files:
        filename = os.path.join(f'outputs/{args.model_name}/{args.dataset}/', file)
        if not(filename.endswith(".jsonl")):
            continue

        predictions, answers = [], []
        name = '.'.join(file.split('.')[:-1])
        with open(f"{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if 'qwen3' in args.model_name:
                    pred = data["pred"].split('<|im_end|>')[0]
                elif 'llama3' in args.model_name:
                    pred = data['pred'].replace('<|eot_id|>','')
                predictions.append(pred)
                 
                answers.append(data["answers"])

        score = scorer(predictions, answers)
        if (name =='full'):
            output.append(f"full\t\t{score}")
        else:
            num = int(name.split('-')[1])
            scores.append( (num, score) )
            
    scores.sort(key=lambda x: x[0], reverse=True)
    
    for num, score in scores:
        output.append(f"{num}\t\t{score}")
    
    with open(f'outputs/{args.model_name}/{args.dataset}/result.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))
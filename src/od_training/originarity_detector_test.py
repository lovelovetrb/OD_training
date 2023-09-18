import os
import time

import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.nn import Softmax
from transformers import BertForSequenceClassification, BertJapaneseTokenizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Tokenizerの準備
tokenizer = BertJapaneseTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
    # "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
)

bert_sc = BertForSequenceClassification.from_pretrained(
    # "./model_sbert"
    "./model_bert_whole_word_masking_same_question"
    # "cl-tohoku/bert-base-japanese-whole-word-masking",
    # num_labels=2,
)

TEST_DATASET_PATH = "data/test_arara_parsed.csv"
# TEST_DATASET_PATH = "data/test_parsed.csv"
df = pd.read_csv(TEST_DATASET_PATH, encoding="utf-8")

questions = df["question"].values
answers = df["answer"].values
labels = df["label"].values


predicted = []
correct_labels = []
wrong = []
i = 0
correct_num = 0
max_length = 512
start_time = time.time()
sf = Softmax(dim=1)

for q, a, label in zip(questions, answers, labels):
    correct_labels.append(label)
    correct = label

    encoding = tokenizer(
        q,
        a,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # encoding = {k: v.cuda() for k, v in encoding.items()}
    encoding = {k: v for k, v in encoding.items()}

    with torch.no_grad():
        output = bert_sc.forward(**encoding)
        scores = output.logits
        labels_predicted = scores[0].argmax(-1).cpu().numpy().tolist()
        # print("AI" if labels_predicted == 1 else "Human")
        # print("AI" if correct == 1 else "Human")
        predicted.append(labels_predicted)

    if labels_predicted == correct:
        correct_num += 1
    else:
        wrong.append([q, a, labels_predicted, correct])
    i += 1
    print(f"accuracy: {correct_num / i }")
    scores = sf(scores)
    print(scores)

end_time = time.time()
print("-" * 20)
print(f"elapsed time: {end_time - start_time}")
# print(f"accuracy: {correct_num / len(questions) }")
print(f"accuracy: {accuracy_score(correct_labels, predicted)}")
print(f"f1 score: {f1_score(correct_labels, predicted) }")
print(f"precision: {precision_score(correct_labels, predicted) }")
print(f"recall: {recall_score(correct_labels, predicted) }")

# print(wrong)

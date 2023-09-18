import logging
import os

import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from BertForSequenceClassification_pl import BertForSequenceClassification_pl
from torch.utils.data import DataLoader, random_split
from transformers import BertJapaneseTokenizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s : %(message)s"
)


def config_sanity_check(config):
    # データセットのパスが存在するか
    if not os.path.exists(config["dataset_path"]):
        raise FileNotFoundError("データセットのパスが存在しません。")

    # データセットの割合が正しいか
    if config["train_data_ratio"] > 1 or config["train_data_ratio"] < 0:
        raise ValueError("train_data_ratioには0~1の値を指定してください。")

    # patienceが正しいか
    if config["patience"] < 0:
        raise ValueError("patienceには0以上の値を指定してください。")

    if config["lr"]:
        try:
            config["lr"] = float(config["lr"])
        except ValueError:
            raise ValueError("lrにはfloat型の値を指定してください。")

    logging.info("config_sanity_check: OK")


# データの読み込み
with open("config.yaml") as file:
    config = yaml.safe_load(file)
    config = config["train"]
    config_sanity_check(config)

DATASET_PATH = config["dataset_path"]
df = pd.read_csv(DATASET_PATH, encoding="utf-8")

questions = df["question"].values
answers = df["answer"].values
labels = df["label"].values


# 1. BERT Tokenizerを用いて単語分割・IDへ変換
logging.info("data tokenize")
# Tokenizerの準備
tokenizer = BertJapaneseTokenizer.from_pretrained(
    # "cl-tohoku/bert-base-japanese-whole-word-masking"
    "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
)

dataset_for_loader = []

# 1文づつ処理
for q, a, label in zip(questions, answers, labels):
    token_words = tokenizer(q, a, max_length=512, padding="max_length", truncation=True)

    # DEBUG
    # print(tokenizer.decode(token_words["input_ids"]))

    token_words["labels"] = label  # ラベルを追加
    token_words = {k: torch.tensor(v) for k, v in token_words.items()}
    dataset_for_loader.append(token_words)

# 80%地点のIDを取得
TRAIN_DATA_RATIO = config["train_data_ratio"]
train_size = int(TRAIN_DATA_RATIO * len(dataset_for_loader))
val_size = len(dataset_for_loader) - train_size

# データセットを分割
train_dataset, val_dataset = random_split(dataset_for_loader, [train_size, val_size])
logging.info("dataset created")

# データローダの作成
dataloader_train = DataLoader(train_dataset, batch_size=16, shuffle=True)
dataloader_val = DataLoader(val_dataset, batch_size=16)

# MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"
MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"


# 2. PyTorch LightningでBERTをfine-tuning
logging.info("fine-tuning start!")
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_weights_only=True,
    dirpath="model/",
)

early_stopping = pl.callbacks.EarlyStopping(
    monitor="val_loss", mode="min", patience=config["patience"]
)

devices = [x for x in range(config["devices"])]

trainer = pl.Trainer(
    accelerator="gpu",
    devices=devices,
    max_epochs=config["max_epochs"],
    callbacks=[checkpoint, early_stopping],
)

model = BertForSequenceClassification_pl(
    MODEL_NAME, num_labels=config["num_labels"], lr=config["lr"]
)

trainer.fit(model, dataloader_train, dataloader_val)

best_model_path = checkpoint.best_model_path  # ベストモデルのファイル
logging.info("ベストモデルのファイル: ", checkpoint.best_model_path)
logging.info("ベストモデルの検証データに対する損失: ", checkpoint.best_model_score)

# PyTorch Lightningモデルのロード
best_model = BertForSequenceClassification_pl.load_from_checkpoint(best_model_path)

# Transformers対応のモデルを./model_transformesに保存
model.save_for_transformers(best_model)
logging.info("bast model saved and finished")

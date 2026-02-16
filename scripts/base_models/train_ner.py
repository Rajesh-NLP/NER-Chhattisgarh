import os
import json
import argparse
import numpy as np
from collections import Counter
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from peft import LoraConfig, get_peft_model
import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True)
parser.add_argument("--tokenizer_name", required=True)
parser.add_argument("--train_file", required=True)
parser.add_argument("--test_file", required=True)
parser.add_argument("--output_dir", required=True)

parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--max_len", type=int, default=128)

parser.add_argument("--rank", type=int, default=32)
parser.add_argument("--alpha", type=int, default=32)
parser.add_argument("--fp16", action="store_true")

parser.add_argument("--model_type", choices=["xlmr", "mbert", "indicbert"], required=True)

args = parser.parse_args()

def read_conll(path):
    sentences, tags = [], []
    words, labels = [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append(words)
                    tags.append(labels)
                    words, labels = [], []
                continue
            parts = line.split()
            words.append(parts[0])
            labels.append(parts[-1])

    if words:
        sentences.append(words)
        tags.append(labels)

    return sentences, tags

train_tokens, train_tags = read_conll(args.train_file)
test_tokens, test_tags   = read_conll(args.test_file)

unique_tags = sorted(
    set(t for s in train_tags for t in s) |
    set(t for s in test_tags for t in s)
)

label2id = {t: i for i, t in enumerate(unique_tags)}
id2label = {i: t for t, i in label2id.items()}

train_ds = Dataset.from_dict({"tokens": train_tokens, "ner_tags": train_tags})
test_ds  = Dataset.from_dict({"tokens": test_tokens,  "ner_tags": test_tags})


tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

def tokenize_and_align(examples):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=args.max_len
    )

    labels = []
    for i in range(len(examples["tokens"])):
        word_ids = tokenized.word_ids(batch_index=i)
        prev = None
        label_ids = []

        for w in word_ids:
            if w is None:
                label_ids.append(-100)
            elif w != prev:
                label_ids.append(label2id[examples["ner_tags"][i][w]])
            else:
                label_ids.append(-100)
            prev = w

        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized

train_ds = train_ds.map(tokenize_and_align, batched=True, remove_columns=train_ds.column_names)
test_ds  = test_ds.map(tokenize_and_align, batched=True, remove_columns=test_ds.column_names)


model = AutoModelForTokenClassification.from_pretrained(
    args.model_name,
    num_labels=len(unique_tags),
    id2label=id2label,
    label2id=label2id
)

model.resize_token_embeddings(len(tokenizer))

if args.model_type == "xlmr":
    target_modules = ["query", "value"]
else:
    target_modules = ["query", "key", "value"]

lora = LoraConfig(
    r=args.rank,
    lora_alpha=args.alpha,
    lora_dropout=0.1,
    target_modules=target_modules,
    bias="none",
    task_type="TOKEN_CLS"
)

model = get_peft_model(model, lora)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    learning_rate=args.lr,
    fp16=args.fp16,
    save_strategy="no",
    logging_steps=50,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    data_collator=DataCollatorForTokenClassification(tokenizer)
)

trainer.train()

metric = evaluate.load("seqeval")
pred = trainer.predict(test_ds)

pred_ids = np.argmax(pred.predictions, axis=-1)

true_preds = [
    [id2label[p] for p, l in zip(ps, ls) if l != -100]
    for ps, ls in zip(pred_ids, pred.label_ids)
]

true_labels = [
    [id2label[l] for p, l in zip(ps, ls) if l != -100]
    for ps, ls in zip(pred_ids, pred.label_ids)
]

results = metric.compute(predictions=true_preds, references=true_labels)

counts = Counter()
for sent in true_labels:
    for tag in sent:
        if tag != "O":
            counts[tag.replace("B-", "").replace("I-", "")] += 1

tag_scores = {}
for k, v in results.items():
    if isinstance(v, dict) and "f1" in v:
        tag_scores[k] = {
            "precision": v["precision"],
            "recall": v["recall"],
            "f1": v["f1"],
            "count": counts.get(k, 0)
        }

os.makedirs(args.output_dir, exist_ok=True)

with open(os.path.join(args.output_dir, "tag_wise_scores.json"), "w") as f:
    json.dump(tag_scores, f, indent=4)

pred_file = os.path.join(args.output_dir, "token_level_predictions.conll")

with open(pred_file, "w", encoding="utf-8") as f:
    for sent_id, (tokens, golds, preds) in enumerate(
        zip(test_tokens, true_labels, true_preds)
    ):
        f.write(f"# sent_id = {sent_id}\n")
        for tok, gold, pred in zip(tokens, golds, preds):
            f.write(f"{tok}\t{gold}\t{pred}\n")
        f.write("\n")

trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

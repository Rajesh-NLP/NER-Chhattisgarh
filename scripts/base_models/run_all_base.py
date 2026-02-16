import os
import subprocess

DATA_ROOT = "/home/intern1/NER-V0.3"
RESULT_ROOT = "/home/intern1/NER-MODELS/finetunning_on_base_model"
SCRIPT_PATH = "/home/intern1/NER-MODELS/scripts/base_models/train_ner.py"

LANGUAGE_MAP = {
    "Baigani": "baigani_12",
    "Halbi": "halbi_12",
    "Kudukh": "kudukh_12"
 }

MODELS = {
    "XLM-Roberta": {
        "model_name": "xlm-roberta-base",
        "tokenizer_name": "xlm-roberta-base",
        "model_type": "xlmr"
    },
    "mBERT": {
        "model_name": "bert-base-multilingual-cased",
        "tokenizer_name": "bert-base-multilingual-cased",
        "model_type": "mbert"
    },
    "IndicBERT": {
        "model_name": "ai4bharat/IndicBERTv2-MLM-only",
        "tokenizer_name": "ai4bharat/IndicBERTv2-MLM-only",
        "model_type": "indicbert"
    }
}


EPOCHS = 12
BATCH_SIZE = 8
LR = 2e-4
MAX_LEN = 128
RANK = 32
ALPHA = 32
FP16 = True


for folder_name, lang_id in LANGUAGE_MAP.items():
    lang_path = os.path.join(DATA_ROOT, folder_name)

    train_file = os.path.join(lang_path, f"{folder_name}.train.conll")
    test_file  = os.path.join(lang_path, f"{folder_name}.test.conll")

    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        print(f"[SKIP] Missing train/test for {folder_name}")
        continue

    for model_name, cfg in MODELS.items():
        output_dir = os.path.join(
            RESULT_ROOT,
            lang_id,
            model_name
        )

        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "python", SCRIPT_PATH,
            "--model_name", cfg["model_name"],
            "--tokenizer_name", cfg["tokenizer_name"],
            "--model_type", cfg["model_type"],
            "--train_file", train_file,
            "--test_file", test_file,
            "--output_dir", output_dir,
            "--epochs", str(EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--lr", str(LR),
            "--max_len", str(MAX_LEN),
            "--rank", str(RANK),
            "--alpha", str(ALPHA)
        ]

        if FP16:
            cmd.append("--fp16")

        print(f"\n[RUN] {lang_id} | {model_name}")
        subprocess.run(cmd, check=True)

import os
import csv
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score


BASE_ROOT = "/home/intern1/NER-MODELS/finetunning_on_pretrained_chhattisgarhi_model/"


def load_conll_file(file_path):
    y_true = []
    y_pred = []

    true_sentence = []
    pred_sentence = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if true_sentence:
                    y_true.append(true_sentence)
                    y_pred.append(pred_sentence)
                    true_sentence = []
                    pred_sentence = []
                continue

            if line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 3:
                continue

            _, gold, pred = parts
            true_sentence.append(gold)
            pred_sentence.append(pred)

        if true_sentence:
            y_true.append(true_sentence)
            y_pred.append(pred_sentence)

    return y_true, y_pred


def save_tagwise_report(y_true, y_pred, save_path):
    report_dict = classification_report(y_true, y_pred, output_dict=True)

    # Extract entity tags only (remove averages)
    entity_tags = [
        tag for tag in report_dict.keys()
        if tag not in ["micro avg", "macro avg", "weighted avg"]
    ]

    entity_tags = sorted(entity_tags)

    total_precision = precision_score(y_true, y_pred)
    total_recall = recall_score(y_true, y_pred)
    total_f1 = f1_score(y_true, y_pred)

    with open(save_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["Tag", "Precision", "Recall", "F1-score", "Support"])

        for tag in entity_tags:
            writer.writerow([
                tag,
                round(report_dict[tag]["precision"], 4),
                round(report_dict[tag]["recall"], 4),
                round(report_dict[tag]["f1-score"], 4),
                report_dict[tag]["support"]
            ])

        writer.writerow([])
        writer.writerow([
            "TOTAL (micro)",
            round(total_precision, 4),
            round(total_recall, 4),
            round(total_f1, 4),
            "-"
        ])


def evaluate_all_models(base_root):
    print("\n=========== CoNLL Entity-Level Evaluation ===========\n")

    for language in os.listdir(base_root):
        language_path = os.path.join(base_root, language)

        if not os.path.isdir(language_path):
            continue

        for model_name in os.listdir(language_path):
            model_path = os.path.join(language_path, model_name)

            if not os.path.isdir(model_path):
                continue

            conll_file = os.path.join(model_path, "token_level_predictions.conll")

            if not os.path.exists(conll_file):
                continue

            y_true, y_pred = load_conll_file(conll_file)

            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            print(f"Language : {language}")
            print(f"Model    : {model_name}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1-score : {f1:.4f}")
            print("-" * 50)

            # Save tag-wise CSV
            save_path = os.path.join(model_path, "tagwise_report.csv")
            save_tagwise_report(y_true, y_pred, save_path)


if __name__ == "__main__":
    evaluate_all_models(BASE_ROOT)

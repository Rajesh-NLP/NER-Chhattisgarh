import os
import csv
from collections import defaultdict

BASE_ROOT = "/home/intern1/NER-MODELS/finetunning_on_base_model/"
OUTPUT_FILE = "/home/intern1/NER-MODELS/finetunning_on_base_model.csv"


def collect_language_models(base_root):
    language_models = defaultdict(dict)

    for language in os.listdir(base_root):
        language_path = os.path.join(base_root, language)

        if not os.path.isdir(language_path):
            continue

        if language.endswith("12"):
            continue

        if not language.endswith("25"):
            continue

        for model_name in os.listdir(language_path):
            model_path = os.path.join(language_path, model_name)

            if not os.path.isdir(model_path):
                continue

            csv_path = os.path.join(model_path, "tagwise_report.csv")

            if os.path.exists(csv_path):
                language_models[language][model_name] = csv_path

    return language_models


def read_tagwise_csv(file_path):
    tag_scores = {}

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if not row or "TOTAL" in row[0]:
                continue

            tag = row[0]
            precision = row[1]
            recall = row[2]
            f1 = row[3]

            tag_scores[tag] = (precision, recall, f1)

    return tag_scores


def main():
    language_models = collect_language_models(BASE_ROOT)

    global_tags = set()
    all_data = {}

    # Read everything
    for language in language_models:
        for model in language_models[language]:
            tag_dict = read_tagwise_csv(language_models[language][model])
            all_data[(language, model)] = tag_dict
            global_tags.update(tag_dict.keys())

    global_tags = sorted(global_tags)
    sorted_keys = sorted(all_data.keys())  # sorted by language then model

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out_file:
        writer = csv.writer(out_file)

        # Header
        header = ["Tag"]
        for (language, model) in sorted_keys:
            header.extend([
                f"{language}_{model}_P",
                f"{language}_{model}_R",
                f"{language}_{model}_F"
            ])
        writer.writerow(header)

        # Rows
        for tag in global_tags:
            row = [tag]

            for key in sorted_keys:
                tag_dict = all_data[key]

                if tag in tag_dict:
                    row.extend(tag_dict[tag])
                else:
                    row.extend(["-", "-", "-"])

            writer.writerow(row)

    print(f"\n? Final hierarchical CSV created: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

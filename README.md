# Multilingual NER for Low-Resource Languages

This repository contains datasets, training scripts, and experimental settings for Named Entity Recognition (NER) across **7 low-resource Indian languages**:

---

## 📁 Repository Structure

```
├── data/
├── scripts/
├── setting_1/
├── setting_2/
└── README.md
```

---

## 📂 `data/`

This directory contains the NER datasets for all seven languages.

* Dataset instances and splits (train/test) are organized per language.
* Each language dataset includes annotated named entities.
* Due to size constraints, only dataset instances are provided here.

---

## 📂 `scripts/`

This folder contains:

* Training scripts
* Evaluation scripts
* Utility scripts for preprocessing and model experimentation

These scripts support experiments conducted under both settings described below.

---

## ⚙️ Experimental Settings

### 🔹 Setting I: Direct Low-Rank Adaptation (LoRA)

In this setting, we apply **Low-Rank Adaptation (LoRA)** for efficient fine-tuning of multilingual transformer models on each language independently.

Models trained under Setting I are stored in:

```
setting_1/
```

For each of the 7 languages, the following models were trained:

* **mBERT**
* **XLM-RoBERTa**
* **IndicBERT**

---

### 🔹 Setting II: CPAP

In this setting, we introduce a **Chhattisgarhi-Pivot Adaptive Pre-training (CPAP)** strategy before downstream NER fine-tuning.

Models trained under Setting II are stored in:

```
setting_2/
```

For each of the 7 languages, the following models were trained:

* **mBERT**
* **XLM-RoBERTa**
* **XLM-RoBERTa**
* **IndicBERT**

---

## 🤖 Trained Models Availability

Due to the large size of the trained models for each language and setting, the model weights are **not directly hosted in this repository**.

📩 The trained models are available **upon reasonable request**.
Please contact the repository maintainer for access.

---

## 📊 Summary

* **Task:** Named Entity Recognition (NER)
* **Languages:** 7 low-resource languages
* **Models:** mBERT, XLM-RoBERTa, IndicBERT
* **Settings:**

  * Direct LoRA Fine-tuning
  * CPAP

---


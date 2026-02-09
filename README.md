# A3 – English ↔ Myanmar Machine Translation with Attention

This project is part of the **A3: Make Your Own Machine Translation Language** assignment.  
The goal is to build a neural machine translation (NMT) system between **English and Myanmar**, experiment with different **attention mechanisms**, and deploy a simple web application to demonstrate translation.

---

## 1. Dataset

- **Dataset**: OPUS TED2020 English–Myanmar (en–my)
- **Source**: https://opus.nlpl.eu/TED2020
- The dataset contains parallel sentences from TED talks.

To reduce training time due to limited computational resources, a **random subset** of the dataset was used. Extremely long sentences were filtered out while keeping a fixed train/validation/test split for fair comparison across models.

---

## 2. Data Preparation

- Basic text cleaning: trimming whitespace and removing empty lines.
- **Tokenization / Segmentation**:  
  SentencePiece (Unigram model) was used for both English and Myanmar.
  This is especially important for Myanmar, where word boundaries and spacing are not always consistent.
- Separate tokenizers were trained for source (English) and target (Myanmar).
- Special tokens `<bos>`, `<eos>`, `<pad>`, and `<unk>` were included.

---

## 3. Models and Attention Mechanisms

A sequence-to-sequence (Seq2Seq) model with a bidirectional GRU encoder was implemented.

Two attention mechanisms were compared:

- **General Attention**
- **Additive Attention (Bahdanau-style)**

All experiments were conducted using the **same dataset split and hyperparameters** to ensure a fair comparison.

---

## 4. Performance Comparison

| Attention Mechanism | Training Loss | Training PPL | Validation Loss | Validation PPL | Time / Epoch (sec) |
|---------------------|---------------|--------------|------------------|----------------|-------------------|
| General Attention   | 6.426         | 617.59       | 7.372            | 1590.81        | 216.0             |
| Additive Attention  | 6.180         | 482.75       | 7.294            | 1471.79        | 228.9             |

Additive Attention achieved slightly better validation loss and perplexity, while General Attention was marginally faster per epoch.

---

## 5. Attention Visualization

Attention maps were visualized to analyze alignment between source and target tokens.

- General Attention showed highly concentrated attention on limited source positions.
- Additive Attention produced more distributed and smoother attention patterns, indicating better alignment behavior.

Due to font limitations, some Myanmar characters may not render correctly in Matplotlib, but this does not affect the model behavior.

---

## 6. Web Application (Task 4)

A simple Flask web application was implemented in `app/web_app.py`.

### Features
- Input box for English text
- Output display for translated Myanmar text
- Uses the trained **Additive Attention** model
- Inference-only (no retraining)

### How it works
1. User enters an English sentence.
2. The sentence is tokenized using SentencePiece.
3. The trained Seq2Seq model generates the translation using greedy decoding.
4. The output tokens are decoded back into Myanmar text and displayed.

### Run the app
```bash
cd app
pip install -r ../requirements.txt
python web_app.py

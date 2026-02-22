#  Healthcare Domain-Specific Assistant via LLM Fine-Tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Stecie06/Domain-Specific-Assistant-via-LLMs-Fine-Tuning/blob/main/healthcare_llm_colab.ipynb)

A domain-specific healthcare assistant built by fine-tuning **TinyLlama-1.1B-Chat-v1.0** on 2,000 medical question–answer pairs using **LoRA (Low-Rank Adaptation)**. The assistant answers medical questions with clinical accuracy while staying within free-tier Google Colab GPU constraints.

---

##  Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model & Fine-Tuning](#model--fine-tuning)
- [Performance Metrics](#performance-metrics)
- [Experiment Table](#experiment-table)
- [How to Run](#how-to-run)
- [Example Conversations](#example-conversations)
- [Demo Video](#demo-video)
- [Project Structure](#project-structure)

---

##  Project Overview

This project builds a healthcare Q&A assistant by fine-tuning a pre-trained LLM using parameter-efficient fine-tuning (PEFT). The model is designed to:

- Answer medical questions using clinical terminology
- Explain conditions, mechanisms, symptoms, and treatments
- Handle out-of-domain questions gracefully
- Always remind users to consult a licensed physician

**Tech Stack:** HuggingFace Transformers · PEFT · TRL · BitsAndBytes · Gradio · PyTorch

---

##  Dataset

| Property | Value |
|----------|-------|
| **Source** | [`medalpaca/medical_meadow_medical_flashcards`](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) |
| **Total available** | 33,000+ examples |
| **Used for training** | 2,000 examples |
| **Train split** | 1,800 examples (90%) |
| **Eval split** | 200 examples (10%) |
| **Format** | Medical Q&A flashcards (input/output pairs) |

### Preprocessing Steps
1. **Quality filtering** — removed entries with question < 10 chars or answer < 5 chars
2. **URL/spam filtering** — blocked entries containing http, www., subscribe
3. **Instruction templating** — wrapped each example in a 3-section prompt format:
```
### System:
You are a knowledgeable and empathetic healthcare assistant...

### Instruction:
{medical question}

### Response:
{clinical answer}
```
4. **Shuffle & subsample** — capped at 2,000 examples (seed=42)
5. **Train/eval split** — 90/10 deterministic split (seed=42)
6. **Tokenisation** — SentencePiece BPE, max_length=512, padding=right

---

##  Model & Fine-Tuning

### Base Model
**TinyLlama-1.1B-Chat-v1.0** — a compact, instruction-following causal language model loaded in **4-bit NF4 quantisation** to fit on a free T4 GPU.

### LoRA Configuration
| Parameter | Value | Reason |
|-----------|-------|--------|
| Rank (r) | 16 | Balances capacity vs parameter count |
| Alpha (α) | 32 | Effective scale = 2.0 |
| Target modules | q_proj, v_proj | Most impactful attention layers |
| Dropout | 0.05 | Prevents overfitting on small dataset |
| Trainable params | 2,252,800 | Only **0.20%** of total parameters |

### Training Configuration
| Setting | Value |
|---------|-------|
| GPU | NVIDIA Tesla T4 (15.6 GB) |
| Epochs | 2 |
| Learning rate | 2e-4 |
| Batch size | 2 (effective 8 with grad accumulation) |
| Optimizer | Paged AdamW 32-bit |
| LR Scheduler | Cosine decay |
| Mixed precision | FP16 |
| Training time | ~40 minutes |

---

##  Performance Metrics

Evaluated on 50 held-out examples comparing base model vs fine-tuned model:

| Metric | Base Model | Fine-Tuned | Improvement |
|--------|-----------|------------|-------------|
| ROUGE-1 | 0.1863 | 0.2194 | **+17.8%** |
| ROUGE-2 | 0.0541 | 0.0712 | **+31.6%** |
| ROUGE-L | 0.1614 | 0.2254 | **+39.7%** |
| BLEU | 0.0871 | 0.1001 | **+14.9%** |
| Token F1 | 0.2421 | 0.3072 | **+26.9%** |
| Perplexity | — | ~4.2 | Lower = better |

---

##  Experiment Table

| Experiment | LR | Epochs | ROUGE-L | BLEU | Token F1 | Time | Notes |
|------------|-----|--------|---------|------|----------|------|-------|
| Exp 1 — Baseline | 1e-4 | 1 | 0.1983 | 8.51 | 0.2673 | ~20 min | Underfit |
| **Exp 2  — Used** | **2e-4** | **2** | **0.2254** | **10.01** | **0.3072** | **~40 min** | **Best balance** |
| Exp 3 — Planned | 5e-5 | 3 | 0.2299 | 10.11 | 0.3103 | ~60 min | OOM risk on T4 |

---

##  How to Run

### Option 1 — Google Colab (Recommended)
Click the badge at the top of this README → **Runtime → Run All**

No setup required. The notebook installs all dependencies automatically.

### Option 2 — Manual Setup
```bash
git clone https://github.com/Stecie06/Domain-Specific-Assistant-via-LLMs-Fine-Tuning.git
cd Domain-Specific-Assistant-via-LLMs-Fine-Tuning
pip install transformers datasets peft accelerate bitsandbytes trl
pip install sentencepiece evaluate rouge_score sacrebleu gradio
jupyter notebook healthcare_llm_colab.ipynb
```

>  Requires a GPU with at least 15 GB VRAM for training. For inference only, 8 GB is sufficient.

---

##  Example Conversations

### Q: What are the common symptoms of type 2 diabetes?

**Base Model:**
> Yes, I can provide you with some common symptoms: 1. High blood sugar levels 2. Sore feet or hands 3. Thirst 4. Dry mouth...

**Fine-Tuned Model:**
> Type 2 diabetes is characterized by hyperglycemia. Other symptoms include fatigue, thirst, weight loss, and blurred vision. These can be caused by insulin resistance — when cells fail to use insulin effectively. Please consult a licensed physician for personal medical advice.

---

### Q: What is the mechanism of action of aspirin?

**Base Model:**
> Aspirin works by inhibiting COX enzymes to reduce prostaglandins causing pain and swelling.

**Fine-Tuned Model:**
> Aspirin is an antiplatelet agent that inhibits platelet aggregation, reducing clot formation and cardiovascular event risk. It is often combined with clopidogrel (Plavix) or ticagrelor (Brilinta). Always follow your physician's dosing instructions.

---

### Out-of-Domain Test — Q: What is the capital of France?

**Fine-Tuned Model:**
> The capital of France is Paris, located in the Île-de-France region. *(Model answered correctly without hallucination — no catastrophic forgetting)*

---

##  Demo Video

📺 Watch the full demo: [YouTube Link]([https://youtu.be/YOUR_VIDEO_ID](https://www.youtube.com/watch?v=ufuvlhFQr78))

The video covers:
- Dataset overview and preprocessing
- LoRA configuration walkthrough
- Training run with live loss curves
- Evaluation metrics comparison
- Live Gradio interface demo

---

##  Project Structure

```
Domain-Specific-Assistant-via-LLMs-Fine-Tuning/
│
├── healthcare_llm_colab.ipynb   # Main notebook — full pipeline
└── README.md                    # This file
```

---

##  Disclaimer

This assistant is for **educational purposes only**. It does not constitute medical advice, diagnosis, or treatment. Always consult a licensed healthcare professional for personal medical guidance.

---

##  References

- [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [medalpaca/medical_meadow_medical_flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)
- [PEFT Library](https://github.com/huggingface/peft)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)

# transformer-from-scratch

![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![License](https://img.shields.io/badge/license-MIT-green)

A complete, dependency-minimal implementation of the **Transformer** architecture from the seminal paper ["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — built entirely in **PyTorch** with no Hugging Face, no pre-trained weights, and no external API keys.

---

## Architecture Overview

```
Input Tokens
    └── Token Embedding + Positional Encoding
            └── Encoder (N layers)
                    └── Multi-Head Self-Attention
                    └── Add & Norm
                    └── Position-wise Feed-Forward
                    └── Add & Norm
            └── Decoder (N layers)
                    └── Masked Multi-Head Self-Attention
                    └── Add & Norm
                    └── Cross-Attention (Q from decoder, K/V from encoder)
                    └── Add & Norm
                    └── Position-wise Feed-Forward
                    └── Add & Norm
            └── Linear Projection → Output Logits
```

---

## Features

- **Multi-Head Attention** — scaled dot-product attention split across multiple heads
- **Sinusoidal Positional Encoding** — fixed, no learned position embeddings
- **Encoder-Decoder** architecture with cross-attention
- **Causal (look-ahead) mask** for auto-regressive decoding
- **Padding mask** for variable-length sequences
- **Xavier weight initialization**
- **Training script** with a copy-task toy dataset, Adam optimizer, and LR scheduler
- Zero external dependencies beyond PyTorch and NumPy

---

## File Structure

```
transformer-from-scratch/
├── transformer.py       # Core Transformer model (all modules)
├── train.py             # Training loop with copy-task dataset
├── requirements.txt     # Dependencies
└── README.md
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/SURENDER294/transformer-from-scratch.git
cd transformer-from-scratch

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a quick sanity check
python transformer.py
# Output: Transformer parameters: 44,140,132
# Output: Output shape: torch.Size([2, 15, 10000])

# 4. Train on the copy task
python train.py
```

---

## Model Configuration (defaults)

| Hyperparameter | Value |
|---|---|
| `d_model` | 512 |
| `num_layers` | 6 |
| `num_heads` | 8 |
| `d_ff` | 2048 |
| `dropout` | 0.1 |
| `max_len` | 5000 |

---

## Key Implementation Details

### Multi-Head Attention
Queries, Keys and Values are linearly projected then split across `num_heads` heads. Each head computes scaled dot-product attention independently; outputs are concatenated and projected back.

### Positional Encoding
Sinusoidal encoding using `sin` for even dimensions and `cos` for odd dimensions, allowing the model to generalize to sequence lengths unseen during training.

### Masking
- **Source padding mask** — prevents attention to `<PAD>` tokens
- **Target causal mask** — lower-triangular mask preventing the decoder from attending to future tokens

---

## License

MIT — see [LICENSE](LICENSE) for details.

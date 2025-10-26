# CENTaUR: Cognitive Embeddings for Natural Understanding & Representation

Repository for:

> Binz, M., & Schulz, E. (2023, October). Turning large language models into cognitive models. In The Twelfth International Conference on Learning Representations.

**Important:** Note that the original code is quite outdated. It is mainly for reference purposes. For the original methodology, see this [tutorial paper](https://osf.io/preprints/psyarxiv/f7stn) and [simplified implementation](https://github.com/Zak-Hussain/LLM4BeSci) using HuggingFace.

---

## Ko-CENTaUR: Modern LLM Evaluation (2025)

**Goal**: Replicate the original CENTaUR methodology using modern Korean-capable LLMs

### Models
- **Qwen2.5-32B-Instruct** (QLoRA fine-tuned)
- **DeepSeek-R1-Distill-Qwen-32B** (QLoRA fine-tuned)

### Evaluation Pipeline

Following the original Binz & Schulz (2023) methodology:

1. **Feature Extraction**: Extract last-layer hidden states (not token probabilities)
2. **100-fold LOO CV**: Nested cross-validation for regularization
3. **Binomial Regression**: Fit cognitive model to predict human choices
4. **NLL Metric**: Compare against baselines (Random ≈ 120K, LLaMA-65B ≈ 30K)

### Quick Start

**Feature Extraction**:
```bash
# Test locally (10 samples)
python scripts/extract_centaur_features.py --model qwen25 --n_samples 10

# Run on server (full dataset)
sbatch scripts/submit_extract_qwen25.sh
sbatch scripts/submit_extract_deepseek.sh
```

**Cross-Validation** (Coming soon):
```bash
python scripts/fit_centaur_loo_cv.py --model qwen25
```

### Project Structure

```
.
├── scripts/
│   ├── extract_centaur_features.py     # Feature extraction (production)
│   ├── fit_centaur_loo_cv.py          # 100-fold LOO CV
│   ├── submit_extract_qwen25.sh       # SLURM: Qwen2.5
│   └── submit_extract_deepseek.sh     # SLURM: DeepSeek-R1
├── legacy/
│   ├── choices13k/                    # Original LLaMA implementation
│   └── models.py                      # BinomialRegression class
├── claudedocs/
│   └── EVALUATION_METHODOLOGY_ANALYSIS.md  # Detailed methodology
└── CLAUDE.md                          # Complete project documentation
```

### Documentation

- **[CLAUDE.md](CLAUDE.md)**: Complete project documentation and workflow
- **[EVALUATION_METHODOLOGY_ANALYSIS.md](claudedocs/EVALUATION_METHODOLOGY_ANALYSIS.md)**: Detailed comparison of original vs. current methodology

### Current Status (2025-10-26)

- ✅ Fine-tuning: Qwen2.5-32B, DeepSeek-R1 trained on Choices13k
- ✅ Feature extraction scripts: Production-ready
- ✅ SLURM infrastructure: Ready for server deployment
- ⏳ LOO CV implementation: In progress
- ⏳ Benchmark results: Pending

### Citation

```bibtex
@inproceedings{binz2023turning,
  title={Turning large language models into cognitive models},
  author={Binz, Marcel and Schulz, Eric},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```

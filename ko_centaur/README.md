# Ko-CENTaUR

Korean Language Adaptation of CENTaUR (Cognitive Experimental Nexus utilized as a Unified Representation)

## Project Overview

Ko-CENTaUR adapts the CENTaUR cognitive modeling framework to Korean language and Korean psychological experiment data, enabling prediction of Korean cognitive and clinical measures from behavioral data.

## Directory Structure

```
ko_centaur/
├── data/                      # Data processing and management
│   ├── raw/                   # Raw data from labs (JSONL)
│   ├── processed/             # Preprocessed data (EXAONE format)
│   └── norms/                 # Korean normative data
├── training/                  # Training scripts and utilities
│   ├── train_qlora.py         # QLoRA fine-tuning
│   ├── config.py              # Training configurations
│   └── utils.py               # Training utilities
├── evaluation/                # Evaluation metrics and reports
│   ├── metrics.py             # Core metrics (NLL, accuracy, etc.)
│   └── reports.py             # Report generation
├── adapters/                  # Dual adapter system
│   ├── public.py              # Public adapter (shareable)
│   └── private.py             # Private adapter (restricted)
├── multimodal/                # Multimodal integration (Phase 3)
│   ├── hybrid_model.py        # EXAONE + Qwen2-VL
│   └── vision.py              # Vision processing
└── models/                    # Trained model weights
```

## Quick Start

### Environment Setup

```bash
conda create -n ko-centaur python=3.10
conda activate ko-centaur
pip install torch==2.5.1 transformers==4.57.0 peft datasets accelerate bitsandbytes
```

### Download Psych-101 Data

```bash
python ko_centaur/data/download_psych101.py
python ko_centaur/data/preprocess_psych101.py
```

### Train Model

```bash
# Test run (50 samples)
python ko_centaur/training/train_exaone_qlora.py \
  --data ko_centaur/data/processed/psych101_exaone_train.jsonl \
  --model LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct \
  --output ko_centaur/models/test \
  --num_samples 50

# Full training (60K samples)
python ko_centaur/training/train_exaone_qlora.py \
  --data ko_centaur/data/processed/psych101_exaone_train.jsonl \
  --model LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct \
  --output ko_centaur/models/exaone-psych101 \
  --epochs 3
```

## Documentation

- [Setup Guide](../docs/SETUP_GUIDE.md) - Complete setup instructions
- [Implementation Workflow](../docs/IMPLEMENTATION_WORKFLOW.md) - 52-week timeline
- [LLM Selection Strategy](../docs/LLM_SELECTION_STRATEGY.md) - Model selection rationale

## Current Status

**Phase 1: MVP Validation (In Progress)**

- [x] Environment setup
- [x] EXAONE-3.0-7.8B integration
- [x] Psych-101 data pipeline
- [x] QLoRA training test (50 samples)
- [ ] Tokenizer benchmark
- [ ] Full training (60K samples)
- [ ] Evaluation framework
- [ ] Korean data collection

## Citation

Based on:
- Binz, M., & Schulz, E. (2023). Using cognitive psychology to understand GPT-3. PNAS.
- Original CENTaUR: https://github.com/marcelbinz/CENTaUR

## License

Apache 2.0 (inherits from EXAONE-3.0)

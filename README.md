# TinyLLM

TinyLLM is a minimal transformer-based language model implementation inspired by [Andrej Karpathy's GPT tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY). This repository is intended for learning and experimenting with transformer architectures and training workflows.

## Features
- Simple transformer model (TinyLLM) for language modeling
- Configurable hyperparameters via command line or JSON
- Training, evaluation, and text generation scripts
- Reproducible experiment tracking (hyperparameters and metrics saved per run)

## Installation
Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
Train a model with custom hyperparameters:
```bash
python3 train.py \
  --epochs 10 \
  --context-len 256 \
  --embedding-size 384 \
  --num-heads 6 \
  --num-layers 4 \
  --dropout 0.1 \
  --batch-size 512 \
  --learning-rate 5e-4 \
  --train-ratio 0.9
```

Or use a JSON file for hyperparameters:
```bash
python3 train.py --hparams-json hparam.json
```
Command-line arguments override values in the JSON file.

### Text Generation
Generate text with a trained model:
```bash
python3 predict.py \
  --checkpoint runs/latest/best_model.pt \
  --data tinyshakespeare.txt \
  --prompt "Once upon a time" \
  --tokens 200 \
  --context-len 256 \
  --embedding-size 384 \
  --num-heads 6 \
  --num-layers 4 \
  --dropout 0.1
```

## Experiment Tracking
- Each training run saves checkpoints, metrics, and hyperparameters in a timestamped directory under `runs/`.
- Hyperparameters are saved as `hparams.json` and metrics as `metrics.json`.

## Reference
- [GPT: Building a Generative Language Model from Scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy
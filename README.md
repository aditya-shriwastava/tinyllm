# TinyLLM

A minimal transformer-based language model for educational and experimental purposes.

---

## Showcase

### Sample Output

Command used:
```bash
python3 predict.py --prompt "Before we proceed any " --tokens 1000
```

<details>
<summary>Generated Output</summary>

```
Before we proceed any poison.  
I think the best day should forget swear't:
For thou, thou dost possess. My feart name be:
She would be prisoner runs as he drunk, for his
mildish. What's to some other of his is knave?

MENENIUS:
He's lucker him: he's capital. I am come to Juliet,
Proclaim to be fury him where objectica,
And I, not he is, he who's not. I am:
Their most any comfort and some play'd
To service the hundred him still. Farewell, which you ere
If you were with you.

POLIXENES:
Howsoeverly in men?

CAMILLO:
Madam, a Pomfret:
Why could you make haste, should buy apack your
magic.

HASTINGS:
He shall after.

LEONTES:
A much deposed by this lady her,
As none base that is the fury world.
Stay thou art Coriolanus!
Thou promisent out o'er the dear to keep your friends,
Divine my heart's name in dyumbles;
And after thing of my brother brother Lancasters;
So had born with them of my thing warm.

CATESBY:
Welcome the hope of thee strength victory;
Than which too rare for in the duke of Marian Oxford?
Therefore
```
</details>

---

## Training Metrics

| Epoch | Train Loss | Validation Loss |
|-------|------------|----------------|
| 1     | 1.595      | 1.484          |
| 2     | 1.103      | 1.604          |
| 3     | 0.928      | 1.754          |
| 4     | 0.806      | 1.918          |
| 5     | 0.720      | 2.051          |

- **Train loss** decreased steadily, indicating the model learned from the training data.
- **Validation loss** increased, which may suggest overfitting or a mismatch between training and validation sets.

---

## Features
- Minimal transformer (TinyLLM) for language modeling
- Configurable hyperparameters (CLI or JSON)
- Training, evaluation, and text generation scripts
- Reproducible experiment tracking (hyperparameters and metrics per run)

---

## Installation
Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### Training
Train with custom hyperparameters:
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

Or use a JSON file:
```bash
python3 train.py --hparams-json hparam.json
```
(Command-line arguments override JSON.)

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

---

## Experiment Tracking
- Each run saves checkpoints, metrics, and hyperparameters under `runs/` (timestamped).
- Hyperparameters: `hparams.json`, Metrics: `metrics.json`.

---

## Reference
- [GPT: Building a Generative Language Model from Scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy
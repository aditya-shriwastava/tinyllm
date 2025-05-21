import torch
from model import TinyLLM
import argparse
from tokenizer import Tokenizer

from tqdm import tqdm

def sample(model, tokenizer, device, start_text='', num_tokens=100, temperature=1.0):
    model.eval()
    input_ids = tokenizer.encode(start_text).to(device).unsqueeze(0)

    generated = input_ids[0].tolist()
    for _ in tqdm(range(num_tokens), desc="Sampling", leave=False):
        input_crop = input_ids[:, -model.block_size:]
        logits, _ = model(input_crop)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        generated.append(next_id.item())
        input_ids = torch.cat([input_ids, next_id], dim=1)
    return tokenizer.decode(generated)

def main():
    parser = argparse.ArgumentParser(description='Sample text from a trained TinyLLM model')
    parser.add_argument('--checkpoint', type=str, default='runs/latest/best_model.pt', help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='tinyshakespeare.txt', help='Path to data file for vocab extraction')
    parser.add_argument('--prompt', type=str, default='', help='Initial text prompt')
    parser.add_argument('--tokens', type=int, default=200, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    args = parser.parse_args()

    # Load tokenizer from data
    with open(args.data, 'r') as f:
        data = f.read()
    tokenizer = Tokenizer(data=data)

    # Model params (ensure these match training)
    block_size = 256
    embedding_size = 64
    head_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = TinyLLM(
        vocab_size=len(tokenizer),
        block_size=block_size,
        embedding_size=embedding_size,
        head_size=head_size
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Generate text
    generated = sample(
        model,
        tokenizer,
        device,
        start_text=args.prompt,
        num_tokens=args.tokens,
        temperature=args.temperature
    )
    print(generated)

if __name__ == '__main__':
    main()
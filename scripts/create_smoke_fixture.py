"""Build a tiny, random, fully local model + tokenizer + Dataset for integration tests."""
import argparse
import json
from pathlib import Path


def create(root, samples=3):
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    vocab = {'[UNK]': 0, '[BOS]': 1, '[EOS]': 2, '[PAD]': 3,
             **{f'w{i}': i+4 for i in range(28)}}
    backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                                       bos_token='[BOS]', eos_token='[EOS]', pad_token='[PAD]',
                                       model_max_length=256)
    tokenizer.save_pretrained(root/'models/local/tiny-gpt2/tokenizer')
    torch.manual_seed(7)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=256, n_embd=16,
                                     n_layer=1, n_head=2, bos_token_id=1, eos_token_id=2, pad_token_id=3))
    model.save_pretrained(root/'models/local/tiny-gpt2/model')
    texts = [' '.join(f'w{(i+j)%20}' for i in range(180)) for j in range(samples)]
    Dataset.from_dict({'text': texts}).save_to_disk(str(root/'datasets/allenai/c4/realnewslike/train'))
    (root/'smoke.toml').write_text('''[experiment]
name = "tiny-offline"
output_root = "runs"
seed = 1
threads = 1

[generation]
model_root = "models"
dataset_root = "datasets"
model = "local/tiny-gpt2"
method = "gumbel"
device = "cpu"
number_of_experiments = 1
batch_size = 1
prompt_tokens = 50
tokens_count = 64
buffer_tokens = 20
watermark_key_length = 8
truncate_vocab = 8

[detection]
rolling_window_size = 20
permutation_count = 3

[segmentation]
backend = "python" # Explicit experimental adapter for the CPU engineering smoke test.
minimum_length = 20
permutation_count = 3
block_size = 4
threshold = 0.25
''')
    (root/'fixture.json').write_text(json.dumps({'model_seed': 7, 'vocab': vocab, 'texts': texts}, indent=2))
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing directories are rejected.')
    args = parser.parse_args()
    print(create(args.output)/'smoke.toml')

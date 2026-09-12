"""One subprocess per stage keeps model memory and logs isolated."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=['generate', 'detect', 'segment'], required=True)
    args = parser.parse_args()
    config = json.loads((args.run_dir/'config.resolved.json').read_text())
    import torch
    torch.set_num_threads(config['experiment']['threads'])
    from .runtime import write_json
    write_json(args.run_dir/args.stage/'runtime.json', {
        'torch': torch.__version__, 'cuda_build': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
        'requested_device': config['generation']['device'],
        'threads': torch.get_num_threads(),
    })
    gen = config['generation']
    prefix = args.run_dir/'generate/sample'
    if args.stage == 'generate':
        from .config import generation_args
        from .generation import run
        run(generation_args(config, prefix))
        # Verify the output contract before marking the stage successful.
        from .detection import read_tokens
        meta = json.loads((Path(gen['model_root'])/gen['model']/'model/config.json').read_text())
        seeds, tokens = read_tokens(prefix, meta['vocab_size'])
        inserted = sum(int(value) for value in gen['insertion_blocks_length'].split(','))
        if not inserted and tokens.shape != (gen['number_of_experiments'], gen['tokens_count']):
            raise ValueError(f'Unexpected generated output shape: {tokens.shape}')
    elif args.stage == 'detect':
        from .detection import detect
        meta = json.loads((Path(gen['model_root'])/gen['model']/'model/config.json').read_text())
        detect(prefix, args.run_dir/'detect', vocab_size=meta['vocab_size'], method=gen['method'],
               watermark_key_length=gen['watermark_key_length'], seed=config['experiment']['seed'],
               **config['detection'])
    else:
        options = dict(config['segmentation'])
        backend = options.pop('backend')
        if backend == 'r':
            import subprocess
            from .runtime import ROOT
            command = ['Rscript', '--vanilla', str(ROOT/'scripts/segment_original.R'),
                       str(args.run_dir/'detect'), str(args.run_dir/'segment'),
                       str(gen['number_of_experiments']), str(gen['tokens_count']),
                       str(config['detection']['rolling_window_size']),
                       str(options['minimum_length']), str(options['permutation_count']),
                       str(options['threshold']), str(config['experiment']['seed'])]
            subprocess.run(command, check=True)
            return
        from .segmentation import segment
        segment(args.run_dir/'detect', args.run_dir/'segment',
                sample_count=gen['number_of_experiments'], token_count=gen['tokens_count'],
                rolling_window_size=config['detection']['rolling_window_size'],
                seed=config['experiment']['seed'], **options)


if __name__ == '__main__':
    main()

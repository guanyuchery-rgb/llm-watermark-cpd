"""python -m cpd check/run --config configs/local.toml"""
import argparse
from pathlib import Path
import sys

from .config import load_config, preflight
from .runtime import ORDER, execute


def main():
    parser = argparse.ArgumentParser(description='Local watermark experiment runner; no Slurm required.')
    parser.add_argument('command', choices=['check', 'run'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['all', *ORDER], default='all')
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == 'check':
            stages = ORDER if args.stage == 'all' else [args.stage]
            prefix = (args.run_dir or Path(config['experiment']['output_root'])/'__preflight__')/'generate/sample'
            vocab = preflight(config, stages, prefix)
            # Only import native/ML modules after inexpensive parameter and path checks.
            import watermarking.gumbel.gumbel_levenshtein
            import watermarking.transform.transform_levenshtein
            import transformers, datasets, accelerate
            import torch
            if 'generate' in stages and config['generation']['device'] == 'cuda' and not torch.cuda.is_available():
                raise ValueError('device=cuda was requested but CUDA is unavailable; check the driver and Torch build.')
            print(f'Preflight passed: vocab_size={vocab}. Model weights were not loaded.')
            print('Usable text count is checked before model loading during generation.')
        else:
            run = execute(config, args.config, args.stage, args.run_dir)
            print(f'Completed {args.stage}: {run}')
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f'Experiment failed: {exc}', file=sys.stderr)
        if isinstance(exc, ImportError):
            print('Install requirements, then run: python 1-setup.py build_ext --inplace', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('Interrupted; run status and completed logs have been preserved.', file=sys.stderr)
        return 130
    return 0


if __name__ == '__main__':
    sys.exit(main())

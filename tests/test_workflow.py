"""CPU-only integration and regression checks, with no model/data downloads."""
import ast
import csv
import json
import hashlib
import os
import platform
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.create_smoke_fixture import create
        cls.temp = tempfile.TemporaryDirectory(prefix='cpd-tests-')
        cls.root = create(Path(cls.temp.name)/'fixture')
        cls.baseline = json.loads((ROOT/'tests/fixtures/generation_baseline.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def command(self, cmd, success=True, cwd=ROOT):
        env = os.environ | {'PYTHON_BIN': sys.executable}
        process = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=180)
        if success:
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        else:
            self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
        return process

    def bash_command(self, count, output):
        return ['bash', str(ROOT/'scripts/run_textgen_smoke.sh'),
                '--model-root', str(self.root/'models'), '--dataset-root', str(self.root/'datasets'),
                '--model', 'local/tiny-gpt2', '--output-dir', str(output), '--method', 'gumbel',
                '--number-of-experiments', str(count), '--batch-size', '1', '--prompt-tokens', '50',
                '--tokens-count', '64', '--buffer-tokens', '20', '--watermark-key-length', '1000',
                '--device', 'cpu']

    def test_bash_baseline_single_and_multiple(self):
        # Entire CSV contents must match the saved pre-refactor fixture on this CPU environment.
        for count in (1, 3):
            output = self.root/f'baseline-{count}'
            command = self.bash_command(count, output)
            self.command(command, cwd=self.root)  # Works outside the repository too.
            for suffix, expected in self.baseline['runs'][str(count)].items():
                with (output/f'smoke-{suffix}.csv').open(newline='') as stream:
                    actual = [[int(value) for value in row] for row in csv.reader(stream)]
                self.assertEqual([len(row) for row in actual], [len(row) for row in expected], suffix)
                if platform.system() == 'Darwin' and platform.machine() == 'arm64':
                    self.assertEqual(actual, expected, suffix)
            before = (output/'smoke-seeds.csv').read_bytes()
            error = self.command(command, success=False)
            self.assertIn('Refusing to overwrite', error.stderr)
            self.assertEqual((output/'smoke-seeds.csv').read_bytes(), before)

    def test_preflight_missing_and_insufficient_data(self):
        from datasets import Dataset
        output = self.root/'failure-output'
        command = self.bash_command(1, output)
        command[command.index('--model-root')+1] = str(self.root/'missing')
        error = self.command(command, success=False)
        self.assertIn('Missing model directory', error.stderr)
        small = self.root/'too-short/allenai/c4/realnewslike/train'
        Dataset.from_dict({'text': ['w1 w2']}).save_to_disk(str(small))
        command = self.bash_command(1, output)
        command[command.index('--dataset-root')+1] = str(self.root/'too-short')
        error = self.command(command, success=False)
        self.assertIn('Insufficient usable data', error.stderr)
        self.assertFalse(list(output.glob('*.csv')))

    def test_complete_pipeline_and_reproducibility(self):
        config = self.root/'smoke.toml'
        self.command([sys.executable, '-m', 'cpd', 'check', '--config', str(config)])
        runs = []
        for _ in range(2):
            result = self.command([sys.executable, '-m', 'cpd', 'run', '--config', str(config)])
            line = next(line for line in result.stdout.splitlines() if line.startswith('Run directory: '))
            run = Path(line.removeprefix('Run directory: '))
            runs.append(run)
            report = json.loads((run/'run.json').read_text())
            self.assertEqual(report['status'], 'succeeded')
            self.assertEqual([r['exit_code'] for r in report['stages'].values()], [0, 0, 0])
            self.assertEqual(len(list((run/'detect').glob('*.csv'))), 64)
            self.assertTrue((run/'segment/seedbs.csv').is_file())
            self.assertTrue((run/'segment/changepoints.csv').is_file())
            self.assertTrue((run/'inputs.json').is_file())
        # Timings differ; every scientific CSV must be byte-identical.
        for path in runs[0].glob('*/*.csv'):
            other = runs[1]/path.relative_to(runs[0])
            self.assertEqual(path.read_bytes(), other.read_bytes(), str(path))

    def test_failure_status_logs_and_config_validation(self):
        from cpd.config import load_config
        bad = self.root/'unknown.toml'
        bad.write_text('[generation]\nbatch_szie = 1\n')
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            load_config(bad)
        config = self.root/'insufficient.toml'
        config.write_text((self.root/'smoke.toml').read_text().replace('number_of_experiments = 1', 'number_of_experiments = 4'))
        result = self.command([sys.executable, '-m', 'cpd', 'run', '--stage', 'generate', '--config', str(config)], success=False)
        line = next(line for line in result.stdout.splitlines() if line.startswith('Run directory: '))
        run = Path(line.removeprefix('Run directory: '))
        report = json.loads((run/'run.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['stages']['generate']['status'], 'failed')
        self.assertIn('Insufficient usable data', (run/'generate/run.log').read_text())
        self.assertFalse((run/'detect').exists())

    def test_explicit_stage_continuation_and_overwrite_protection(self):
        config = self.root/'smoke.toml'
        result = self.command([sys.executable, '-m', 'cpd', 'run', '--stage', 'generate', '--config', str(config)])
        run = Path(next(line for line in result.stdout.splitlines() if line.startswith('Run directory: ')).removeprefix('Run directory: '))
        for stage in ['detect', 'segment']:
            self.command([sys.executable, '-m', 'cpd', 'run', '--stage', stage,
                          '--config', str(config), '--run-dir', str(run)])
        original = (run/'run.json').read_bytes()
        self.command([sys.executable, '-m', 'cpd', 'run', '--stage', 'segment',
                      '--config', str(config), '--run-dir', str(run)], success=False)
        self.assertEqual((run/'run.json').read_bytes(), original)

    def test_detection_single_sample_and_invalid_index(self):
        from cpd.detection import read_tokens, detect
        root = self.root/'detect-contract'
        root.mkdir()
        prefix = root/'sample'
        (root/'sample-seeds.csv').write_text('123\n')
        (root/'sample-attacked-tokens.csv').write_text('4,5,6,7\n')
        seeds, tokens = read_tokens(prefix, 32)
        self.assertEqual(seeds.shape, (1,))
        self.assertEqual(tokens.shape, (1, 4))
        with self.assertRaisesRegex(ValueError, 'sample index'):
            detect(prefix, root/'out', vocab_size=32, method='gumbel', watermark_key_length=8,
                   rolling_window_size=2, permutation_count=1, sample_index=1)

    def test_seedbs_matches_legacy_python_functions(self):
        import numpy as np
        from cpd import seedbs
        tree = ast.parse((ROOT/'4.1-seedbs.py').read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        namespace = dict(seedbs.__dict__)
        namespace['significance_permutation_count'] = 3
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<legacy-seedbs>', 'exec'), namespace)
        values = [0.1]*15 + [0.9]*15
        random.seed(1)
        expected = namespace['segment_significance'](values)
        random.seed(1)
        actual = seedbs.segment_significance(values, significance_permutation_count=3, block_size=10)
        self.assertEqual(actual, expected)
        np.testing.assert_array_equal(seedbs.get_seeded_intervals(44, unique_int=True),
                                      namespace['get_seeded_intervals'](44, unique_int=True))

    def test_original_algorithm_sources_unchanged(self):
        expected = json.loads((ROOT/'tests/fixtures/algorithm_sources.json').read_text())
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), digest,
                             f'Algorithm source changed: {name}; scientific review is required.')

    def test_detection_statistics_match_original_both_methods(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        from cpd import detection
        original = (ROOT/'tests/fixtures/detection_statistics.py').read_text()
        for method in ['gumbel', 'transform']:
            namespace = dict(detection.__dict__)
            namespace['args'] = SimpleNamespace(method=method, gamma=.4)
            exec(compile(original, '<original-detection-statistics>', 'exec'), namespace)
            values = []
            for stats in [namespace['test_stats'], detection.make_statistics(method, .4)]:
                torch.manual_seed(19)
                values.append(detection.sliding_permutation_test(
                    np.arange(64, dtype=np.int64) % 24, 32, 8, 20, 3, 123, 20, stats))
            np.testing.assert_array_equal(values[0], values[1], err_msg=method)

    def test_original_r_backend_never_silently_falls_back(self):
        from unittest.mock import patch
        from cpd.config import load_config, preflight
        config = load_config(self.root/'smoke.toml')
        config['segmentation'].update(backend='r', block_size=10)
        with patch('shutil.which', return_value=None):
            with self.assertRaisesRegex(ValueError, 'requires Rscript'):
                preflight(config, ['generate', 'detect', 'segment'], self.root/'r-missing/sample')

    def test_not_selects_shortest_and_discards_overlapping_intervals(self):
        from cpd.segmentation import select_not
        rows = [dict(segment_length=40, change_point_index=20, significance=.01, **{'from': 1, 'to': 41}),
                dict(segment_length=10, change_point_index=18, significance=.01, **{'from': 12, 'to': 22}),
                dict(segment_length=10, change_point_index=48, significance=.01, **{'from': 43, 'to': 53}),
                dict(segment_length=8, change_point_index=70, significance=.5, **{'from': 66, 'to': 74})]
        self.assertEqual([r['change_point_index'] for r in select_not(rows, .05)], [18, 48])


if __name__ == '__main__':
    unittest.main()

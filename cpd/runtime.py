"""Run directories, provenance, subprocess logs and explicit stage state."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
ORDER = ['generate', 'detect', 'segment']


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_files():
    files = set()
    for folder in ('cpd', 'watermarking', 'scripts'):
        for path in (ROOT/folder).rglob('*'):
            if path.suffix in ('.py', '.pyx', '.sh', '.R', '.json'):
                files.add(path)
    files.update(ROOT/name for name in ('1-setup.py', '2-textgen.py', '3-detect.py',
                                      '4.1-seedbs.py', '4-seedbs.R', '5-not.R', 'requirements.txt',
                                      'requirements-macos-py312.lock'))
    return sorted(files)


def source_identity():
    return {str(p.relative_to(ROOT)): sha256(p) for p in source_files()}


def environment():
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))}


def input_manifest(config):
    gen = config['generation']
    roots = [Path(gen['model_root'])/gen['model'],
             Path(gen['dataset_root'])/'allenai/c4/realnewslike/train']
    entries = []
    for root in roots:
        if root.exists():
            for path in sorted(p for p in root.rglob('*') if p.is_file()):
                stat = path.stat()
                item = {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
                # Full content hashes of weights and Arrow files identify the actual inputs.
                item['sha256'] = sha256(path)
                entries.append(item)
    return entries


def git_info():
    def call(*args):
        result = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {'commit': call('rev-parse', 'HEAD'), 'status': call('status', '--short')}


def execute(config, config_path, stage, run_dir=None):
    from .config import preflight
    stages = ORDER if stage == 'all' else [stage]
    if run_dir is None:
        if stages[0] != 'generate':
            raise ValueError('detect/segment require --run-dir from a successful previous stage.')
        name = config['experiment']['name']
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run = Path(config['experiment']['output_root'])/f'{name}-{stamp}-{uuid4().hex[:8]}'
        preflight(config, stages, run/'generate/sample')
        run.mkdir(parents=True, exist_ok=False)
        (run/'config.toml').write_bytes(Path(config_path).read_bytes())
        write_json(run/'config.resolved.json', config)
        identity = source_identity()
        for rel in identity:
            destination = run/'source'/rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT/rel).read_bytes())
        write_json(run/'environment.json', environment())
        write_json(run/'source.json', identity)
        manifest = {'started_at': now(), 'finished_at': None, 'status': 'created',
                    'git': git_info(), 'run_dir': str(run), 'stages': {}}
        write_json(run/'run.json', manifest)
    else:
        run = Path(run_dir).resolve()
        manifest = json.loads((run/'run.json').read_text())
        if config != json.loads((run/'config.resolved.json').read_text()):
            raise ValueError('Configuration differs from this run; create a new run for changed parameters.')
        if source_identity() != json.loads((run/'source.json').read_text()):
            raise ValueError('Source code changed since this run; create a new run to keep provenance consistent.')
        if environment() != json.loads((run/'environment.json').read_text()):
            raise ValueError('Python environment changed since this run; create a new run.')
        preflight(config, stages, run/'generate/sample')
    # Refusing an invalid continuation must not change an already successful run's status.
    for current in stages:
        if (run/current).exists():
            raise FileExistsError(f'Stage directory already exists: {run/current}. Start a new run; no files were changed.')
    first = ORDER.index(stages[0])
    if first and manifest['stages'].get(ORDER[first-1], {}).get('status') != 'succeeded':
        raise ValueError(f'{stages[0]} requires successful {ORDER[first-1]} in this run.')
    lock = run/'.running'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(f'Run is locked: {lock}. Check its PID; remove a stale lock only after confirming no task is running.') from exc
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    print(f'Run directory: {run}', flush=True)
    try:
        print('Hashing local inputs for provenance (large weights/datasets may take time).', flush=True)
        current_inputs = input_manifest(config)
        if (run/'inputs.json').exists():
            if current_inputs != json.loads((run/'inputs.json').read_text()):
                raise ValueError('Input files changed since the previous stage; create a new run.')
        else:
            write_json(run/'inputs.json', current_inputs)
        for current in stages:
            index = ORDER.index(current)
            if index and manifest['stages'].get(ORDER[index-1], {}).get('status') != 'succeeded':
                raise ValueError(f'{current} requires successful {ORDER[index-1]} in this run.')
            if index:
                previous = manifest['stages'][ORDER[index-1]]
                for rel, expected in previous.get('outputs', {}).items():
                    path = run/rel
                    if not path.is_file() or sha256(path) != expected:
                        raise ValueError(f'Previous-stage output changed or is missing: {path}')
            target = run/current
            if target.exists():
                raise FileExistsError(f'Stage directory already exists: {target}. Existing results are never overwritten; start a new run.')
            target.mkdir()
            entry = {'status': 'running', 'started_at': now()}
            manifest['stages'][current] = entry
            manifest.update(status='running', finished_at=None)
            write_json(run/'run.json', manifest)
            command = [sys.executable, '-u', '-m', 'cpd.worker', '--run-dir', str(run), '--stage', current]
            entry['command'] = command
            env = os.environ.copy()
            if config['generation']['dataset_source'] == 'local':
                env.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            env.update(OMP_NUM_THREADS=str(config['experiment']['threads']),
                       MKL_NUM_THREADS=str(config['experiment']['threads']), TOKENIZERS_PARALLELISM='false')
            child = None
            try:
                with (target/'run.log').open('x') as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, text=True, start_new_session=True)
                    for line in child.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end='', flush=True)
                    code = child.wait()
                entry.update(exit_code=code, finished_at=now(), status='succeeded' if code == 0 else 'failed')
                entry['outputs'] = {str(p.relative_to(run)): sha256(p) for p in sorted(target.glob('*.csv'))}
                if code:
                    raise RuntimeError(f'{current} exited with code {code}; full log: {target / "run.log"}')
            except BaseException:
                if child is not None and child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                entry.update(status='failed', finished_at=now())
                raise
            finally:
                write_json(run/'run.json', manifest)
        manifest.update(status='succeeded' if stage == 'all' or stage == 'segment' else 'ready', finished_at=now())
        return run
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        error=str(exc), finished_at=now())
        raise
    finally:
        write_json(run/'run.json', manifest)
        lock.unlink(missing_ok=True)

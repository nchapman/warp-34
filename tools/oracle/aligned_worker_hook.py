"""Scoped source instrumentation for one standalone benchmark process.

Restoring files does not undo hooks in already imported Python modules. Do not
reuse this interpreter for other serving work or run concurrent package users.
"""
from contextlib import contextmanager
import json
import os
import time
from pathlib import Path


DFLASH_SOURCE_SHA256 = '75d064936b54db9062de80e9d1c2a25ae01d5fc69b0885e9c265b25c8ce3a399'

def emit(event):
    event.update(pid=os.getpid(), monotonic_ns=time.monotonic_ns())
    with open(os.environ['ORACLE_EVENTS'], 'a') as f:
        f.write(json.dumps(event) + '\n')


def install_config(cls):
    # 0.5.19 exposes one global dtype, not a separate draft compute dtype.
    # Override only the draft ModelConfig constructor argument, before loading.
    original = cls.__init__
    def init(self, *args, **kwargs):
        if kwargs.get('is_draft_model'):
            kwargs['dtype'] = 'bfloat16'
        original(self, *args, **kwargs)
    cls.__init__ = init


def install_worker(cls):
    from functools import wraps
    original_ingest = cls._append_target_hidden_to_draft_kv_by_loc
    @wraps(original_ingest)
    def ingest(self, *args, **kwargs):
        import torch
        with torch.cuda.nvtx.range('w34.proposer'):
            return original_ingest(self, *args, **kwargs)
    cls._append_target_hidden_to_draft_kv_by_loc = ingest
    original = cls.forward_batch_generation
    def forward(self, batch, *args, **kwargs):
        import torch
        prefill = batch.forward_mode.is_extend() or batch.is_extend_in_batch
        idle = batch.forward_mode.is_idle()
        rids = [r.rid for r in batch.reqs]
        if not getattr(self, '_oracle_logged', False):
            runners = {'target': self.target_worker.model_runner,
                       'draft': self.draft_model_runner}
            emit(dict(kind='dtypes', runners={k: dict(
                config_dtype=str(v.model_config.dtype),
                kv_dtype=str(v.kv_cache_dtype),
                parameter_dtypes=sorted({str(p.dtype) for p in v.model.parameters()}))
                for k, v in runners.items()}))
            self._oracle_logged = True
        start = time.monotonic_ns()
        if not prefill and not idle:
            torch.cuda.nvtx.range_push('oracle.decode.round:' + ','.join(rids))
        try:
            result = original(self, batch, *args, **kwargs)
            event = dict(kind='prefill' if prefill else 'idle' if idle else 'decode',
                         rids=rids, start_ns=start)
            if prefill:
                event.update(prefix_lens=list(batch.prefix_lens), extend_lens=list(batch.extend_lens))
            elif not idle:
                # Copy is intentionally synchronizing: timings include telemetry overhead.
                event['commit_lens'] = result.accept_lens.cpu().tolist()
            emit(event)
            return result
        finally:
            if not prefill and not idle:
                torch.cuda.nvtx.range_pop()
                # Engine shutdown terminates spawned workers; flush CUPTI buffers
                # before that, including graph nodes and the completed NVTX range.
                # With nsys capture-range=none this flushes, not stops collection.
                if os.environ.get('ORACLE_PROFILE_FLUSH') == '1':
                    torch.cuda.profiler.stop()
    cls.forward_batch_generation = forward


@contextmanager
def inject(root):
    """Restore exact source bytes on exit, including partial injection failure."""
    originals = {}
    try:
        for relative, line in [
            ('srt/models/dflash.py', 'from aligned_worker_hook import install_dtype_boundary; install_dtype_boundary(DFlashDraftModel)'),
            ('srt/configs/model_config.py', 'from aligned_worker_hook import install_config; install_config(ModelConfig)'),
            ('srt/speculative/dflash_worker_v2.py', 'from aligned_worker_hook import install_worker; install_worker(DFlashWorkerV2)'),
        ]:
            p = Path(root) / relative
            original = p.read_bytes()
            originals[p] = original
            text = original.decode('utf-8')
            if relative == 'srt/speculative/dflash_worker_v2.py':
                import hashlib
                expected = DFLASH_SOURCE_SHA256
                if hashlib.sha256(original).hexdigest() != expected:
                    raise ValueError('unsupported DFlash source hash for proposer patch')
                begin = '        # --- 1) Draft a fixed block with the draft model.\n'
                end = '        # --- 2) Target verify.\n'
                if text.count(begin) != 1 or text.count(end) != 1:
                    raise ValueError('ambiguous draft scope')
                a, b = text.index(begin), text.index(end)
                block = text[a:b]
                text = (text[:a] + "        with torch.cuda.nvtx.range('w34.proposer'):\n"
                        + ''.join('    ' + line if line.strip() else line
                                  for line in block.splitlines(keepends=True)) + text[b:])
            if line not in text:
                p.write_bytes((text + '\n' + line + '\n').encode('utf-8'))
        yield
    finally:
        # Attempt every restoration even if one file has become unwritable.
        errors = []
        for p, original in originals.items():
            try:
                p.write_bytes(original)
            except OSError as exc:
                errors.append(exc)
        if errors:
            raise errors[0]


def normalize_draft(source, destination):
    """Translate speculators' nested config to SGLang's native config schema.

    Weight files remain symlinks to the exact pinned snapshot; no tensor edits.
    """
    source, destination = Path(source).resolve(), Path(destination)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"refusing to reuse nonempty draft destination: {destination}")
    raw = json.loads((source / 'config.json').read_text())
    if raw.get('architectures') != ['DFlash2DraftModel']:
        raise ValueError('require DFlash2DraftModel architecture')
    cfg = dict(raw['transformer_layer_config'])
    cfg.update(architectures=raw['architectures'], dtype=raw['dtype'],
               tie_word_embeddings=False, block_size=raw['block_size'])
    cfg['dflash_config'] = {k: raw[k] for k in (
        'conv_kernel_size', 'conv_group_size', 'selector_rank', 'selector_top_k',
        'mask_token_id', 'sample_from_anchor')}
    cfg['dflash_config']['target_layer_ids'] = raw['aux_hidden_state_layer_ids']
    destination.mkdir(parents=True, exist_ok=True)
    for p in source.iterdir():
        if p.name != 'config.json':
            (destination / p.name).symlink_to(p)
    (destination / 'config.json').write_text(json.dumps(cfg, indent=2))
    return cfg


def install_dtype_boundary(cls):
    """Explicit FP16 target -> BF16 draft boundary for mixed compute dtypes."""
    original = cls.project_target_hidden
    def project(self, hidden):
        return original(self, hidden.to(self.fc.weight.dtype))
    cls.project_target_hidden = project
    forward = cls.forward
    def run(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        if input_embeds is None:
            embedding = self.embed_tokens if self.embed_tokens is not None else self.forward_embed
            input_embeds = embedding(input_ids)
        return forward(self, input_ids, positions, forward_batch,
                       input_embeds=input_embeds.to(self.fc.weight.dtype), **kwargs)
    cls.forward = run

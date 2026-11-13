"""Encode the original chosen/rejected contrasts with a pinned local model.

This command makes no paid API calls. Run model encoding on the approved pod;
preparation tests use a fixed transform and do not download encoder weights.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback

import numpy as np


def encode_contrasts(encoder, corpora, batch_size=64):
    if not corpora or not corpora[0] or len({len(rows) for rows in corpora}) != 1:
        raise ValueError('all traits must have the same nonzero pair count')
    arrays = {}
    for side in ('chosen', 'rejected'):
        texts = [row[side] for rows in corpora for row in rows]
        encoded = np.asarray(encoder.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                            normalize_embeddings=False, show_progress_bar=False))
        if encoded.ndim != 2 or encoded.shape[0] != len(texts) or not np.isfinite(encoded).all():
            raise ValueError('encoder returned invalid embeddings')
        arrays[side] = encoded.reshape(len(corpora), len(corpora[0]), -1)
    if arrays['chosen'].shape != arrays['rejected'].shape:
        raise ValueError('chosen and rejected embedding dimensions differ')
    arrays['contrasts'] = (arrays['chosen'].astype(np.float64) - arrays['rejected']).mean(axis=1)
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--panel', type=Path, default=here / 'panel.json')
    parser.add_argument('--checkpoint', type=Path, default=here / 'embedding_checkpoint.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('use a new output folder; saved embeddings cannot be overwritten')
    args.output.mkdir(parents=True)
    try:
        from sentence_transformers import SentenceTransformer
        checkpoint = json.loads(args.checkpoint.read_text())
        panel = json.loads(args.panel.read_text())
        paths = [args.data_root / (r['slug'] + '.jsonl') for r in panel['traits']]
        corpora = [[json.loads(line) for line in path.read_text().splitlines()] for path in paths]
        prompts = [row['prompt'] for row in corpora[0]]
        if len(prompts) != 445 or len(set(prompts)) != 445:
            raise ValueError('expected 445 distinct prompts')
        if any([row['prompt'] for row in rows] != prompts for rows in corpora):
            raise ValueError('ordered prompt pool differs between traits')
        receipt_path = args.data_root.parents[2] / 'data_receipt.json'
        receipt = json.loads(receipt_path.read_text())
        if receipt['errors'] or receipt['revision'] != panel['data_revision']:
            raise ValueError('data receipt is not valid for this panel')
        known = {Path(r['path']).name: r['sha256'] for r in receipt['files']}
        hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        if any(digest != known[Path(path).name] for path, digest in hashes.items()):
            raise ValueError('data bytes differ from verified release')
        model = SentenceTransformer(checkpoint['repo'], revision=checkpoint['revision'], device=args.device)
        # SentenceTransformer otherwise truncates silently at its configured length.
        texts = [row[side] for rows in corpora for row in rows for side in ('chosen', 'rejected')]
        lengths = model.tokenizer(texts, truncation=False, padding=False, return_length=True)['length']
        if max(lengths) > model.max_seq_length:
            raise ValueError(f'encoder length {max(lengths)} exceeds {model.max_seq_length}; no truncation authorized')
        arrays = encode_contrasts(model, corpora)
        for name, array in arrays.items():
            np.save(args.output / (name + '.npy'), array, allow_pickle=False)
        metadata = {'schema_version': 1, 'checkpoint': checkpoint, 'panel': panel, 'data_sha256': hashes,
                    'receipt_sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                    'command': sys.argv, 'device': args.device, 'max_length': max(lengths),
                    'software': {x: importlib.metadata.version(x) for x in
                                 ['sentence-transformers', 'transformers', 'tokenizers', 'numpy']},
                    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'prediction_scope': 'embeddings only; fit ridge on training traits in maps.py'}
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    except BaseException as error:
        (args.output / 'error.json').write_text(json.dumps({'error': str(error), 'traceback': traceback.format_exc()}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()

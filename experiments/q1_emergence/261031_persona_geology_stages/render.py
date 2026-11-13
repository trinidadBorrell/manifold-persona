"""Freeze original pairs, matched controls, and exact stage-specific model inputs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PREFLIGHT = REPO / 'output/261031_persona_geology_stages/preflight'
DATA_REVISION = '80b7bd1c42c09248ce44b26d993e92f25e854941'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_panel(metadata):
    cells = defaultdict(list)
    words = set()
    for row in metadata:
        word = row['trait'].casefold()
        if word in words or row['keyed'] not in ('+', '-'):
            raise ValueError('duplicate trait or unknown polarity')
        words.add(word)
        cells[row['factor'], row['keyed']].append(row)
    if len(cells) != 10 or len({k[0] for k in cells}) != 5:
        raise ValueError('need five factors and both poles')
    panel = []
    for key, rows in sorted(cells.items()):
        if len(rows) < 2:
            raise ValueError(f'need two traits in cell {key}')
        for row in sorted(rows, key=lambda r: r['trait'].casefold())[:2]:
            panel.append(dict(row, slug=re.sub(r'[-\s]+', '_', row['trait'].lower())))
    return panel


def control_rows(rows, arm, seed=0):
    result = copy.deepcopy(rows)
    if arm == 'scrambled':
        indices = np.random.default_rng(seed).choice(len(rows), len(rows) // 2, replace=False)
        for i in indices:
            result[i]['chosen'], result[i]['rejected'] = result[i]['rejected'], result[i]['chosen']
    elif arm not in ('real', 'second_seed', 'permuted'):
        raise ValueError(f'unknown arm {arm}')
    return result


def render_pair(tokenizer, row, stage, max_length=1024):
    if stage not in ('base', 'sft', 'dpo', 'instruct'):
        raise ValueError(f'unknown stage {stage}')
    if any(not isinstance(row[k], str) or not row[k].strip() for k in ('prompt', 'chosen', 'rejected')):
        raise ValueError('empty or non-text input')
    if row['chosen'] == row['rejected']:
        raise ValueError('identical chosen and rejected')
    special = set(tokenizer.all_special_tokens) | {'<|endoftext|>', '<|user|>', '<|assistant|>', '<|system|>'}
    if any(s in row[k] for s in special for k in ('prompt', 'chosen', 'rejected')):
        raise ValueError('special token in raw preference text')
    if tokenizer.eos_token is None:
        raise ValueError('missing end-of-text token')
    if stage == 'base':
        if tokenizer.eos_token != '<|endoftext|>':
            raise ValueError('base end-of-text token differs from design')
        prefix = '<|endoftext|>' + row['prompt'] + '\n\n'
    else:
        prefix = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                                              tokenize=False, add_generation_prompt=True)
    output = {k: v for k, v in row.items() if k not in ('chosen', 'rejected')}
    for side in ('chosen', 'rejected'):
        if stage == 'base':
            text = prefix + row[side] + tokenizer.eos_token
        else:
            text = tokenizer.apply_chat_template([
                {'role': 'user', 'content': row['prompt']},
                {'role': 'assistant', 'content': row[side]},
            ], tokenize=False, add_generation_prompt=False)
        if not text.startswith(prefix):
            raise ValueError('full chat rendering does not preserve generation prefix')
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded['input_ids'], encoded['offset_mapping']
        if len(ids) > max_length:
            raise ValueError(f'length {len(ids)} exceeds {max_length}')
        boundary = len(prefix)
        answer_indices = [i for i, (start, end) in enumerate(offsets) if end > boundary]
        if not answer_indices:
            raise ValueError('empty tokenized answer')
        first = answer_indices[0]
        crossing = offsets[first][0] < boundary
        if crossing and prefix[offsets[first][0]:].strip():
            raise ValueError('answer token crosses non-whitespace prompt boundary')
        ends = [i for i in range(first, len(ids)) if ids[i] == tokenizer.eos_token_id]
        if not ends:
            raise ValueError('missing answer end-of-text token')
        last = ends[0]
        mask = [first <= i <= last for i in range(len(ids))]
        output[side] = {'text': text, 'input_ids': ids, 'completion_mask': mask,
                        'answer_char_start': boundary, 'boundary_whitespace_shared': crossing}
    return output


def freeze_controls(panel):
    rng = np.random.default_rng(3102028)
    names = [r['slug'] for r in panel]
    permutation = rng.permutation(len(names))
    while np.any(permutation == np.arange(len(names))):
        permutation = rng.permutation(len(names))
    return {'schema_version': 1, 'permuted_data_trait': dict(zip(names, [names[i] for i in permutation])),
            'scramble_seeds': {s: int.from_bytes(hashlib.sha256(('scramble:' + s).encode()).digest()[:8], 'big')
                               for s in names},
            'initialization_seeds': {'real': 0, 'scrambled': 0, 'permuted': 0, 'second_seed': 1},
            'order_seed': 0}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def prepare_pilot_inputs(source, panel, controls, out):
    """Materialize all arms; the stored data-label mapping is the treatment."""
    source, out = Path(source), Path(out)
    names = {r['slug'] for r in panel}
    mapping = controls['permuted_data_trait']
    if set(mapping) != names or set(mapping.values()) != names or any(k == v for k, v in mapping.items()):
        raise ValueError('permuted arm must use a complete derangement')
    records = []
    for arm in ('real', 'scrambled', 'permuted', 'second_seed'):
        for trait in panel:
            slug = trait['slug']
            data_slug = mapping[slug] if arm == 'permuted' else slug
            obj = json.loads((source / (data_slug + '.json')).read_text())
            obj['trait'] = trait
            obj['control'] = {'arm': arm, 'data_trait': data_slug,
                              'initialization_seed': controls['initialization_seeds'][arm],
                              'order_seed': controls['order_seed']}
            if arm == 'scrambled':
                obj['control']['scramble_seed'] = controls['scramble_seeds'][slug]
            obj['rows'] = control_rows(obj['rows'], arm, seed=controls['scramble_seeds'][slug])
            path = out / arm / (slug + '.json')
            write_json(path, obj)
            records.append({'arm': arm, 'trait': slug, 'data_trait': data_slug,
                            'path': str(path), 'sha256': sha256(path)})
    write_json(out / 'manifest.json', {'schema_version': 1, 'files': records})
    return records


def prepare_activation_inputs(tokenizer, bank, training_prompts):
    if set(row['prompt'] for row in bank['rows']).intersection(training_prompts):
        raise ValueError('activation prompt overlaps training prompts')
    rows = []
    for row in bank['rows']:
        rendered = render_pair(tokenizer, dict(row, chosen=row['completion'],
                               rejected='An unused alternative for the renderer.'), 'base')['chosen']
        for i, token in enumerate(rendered['input_ids']):
            if token == tokenizer.eos_token_id:
                rendered['completion_mask'][i] = False
        rows.append(dict(row, **rendered))
    return dict(bank, rows=rows, format='base plain text, identical IDs and masks for every stage')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=PREFLIGHT / 'rendered')
    parser.add_argument('--stages', default='base,sft,dpo,instruct')
    args = parser.parse_args()
    from transformers import AutoTokenizer

    metadata_path = PREFLIGHT / 'adapter_release/traits_primary.json'
    panel = select_panel(json.loads(metadata_path.read_text()))
    controls = freeze_controls(panel)
    write_json(HERE / 'panel.json', {'schema_version': 1, 'data_revision': DATA_REVISION,
                                    'metadata_sha256': sha256(metadata_path), 'traits': panel})
    write_json(HERE / 'controls.json', controls)
    entries = json.loads((HERE.parent / '261029_fixed_text_read_through/checkpoints.json').read_text())['entries']
    checkpoints = {x['name']: x for x in entries}
    data_root = PREFLIGHT / 'results_release/qwen35/data_common'
    raw = {r['slug']: [json.loads(s) for s in (data_root / (r['slug'] + '.jsonl')).read_text().splitlines()]
           for r in panel}
    receipt = json.loads((PREFLIGHT / 'data_receipt.json').read_text())
    if receipt['errors'] or receipt['revision'] != DATA_REVISION:
        raise ValueError('unverified preference data')
    known = {Path(f['path']).name: f['sha256'] for f in receipt['files']}
    for slug in raw:
        if sha256(data_root / (slug + '.jsonl')) != known[slug + '.jsonl']:
            raise ValueError('preference data changed after verification')
    examples, records = [], []
    # First positive word of each factor: five distinct traits, frozen by the panel rule.
    read_traits = [next(r['slug'] for r in panel if r['factor'] == f and r['keyed'] == '+')
                   for f in sorted({r['factor'] for r in panel})]
    for stage in args.stages.split(','):
        checkpoint = checkpoints[stage]
        tok = AutoTokenizer.from_pretrained(checkpoint['repo'], revision=checkpoint['revision'])
        config_digest = hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest()
        for trait in panel:
            slug = trait['slug']
            rows = [render_pair(tok, r, stage) for r in raw[slug]]
            obj = {'schema_version': 1, 'stage': stage, 'trait': trait,
                   'model': {'repo': checkpoint['repo'], 'revision': checkpoint['revision']},
                   'tokenizer': {'repo': checkpoint['repo'], 'revision': checkpoint['revision'],
                                 'backend_sha256': config_digest, 'chat_template': tok.chat_template,
                                 'transformers': importlib.metadata.version('transformers'),
                                 'tokenizers': importlib.metadata.version('tokenizers')},
                   'data': {'repo': 'EternalRecursion/persona-curvature-results', 'revision': DATA_REVISION,
                            'source_path': 'qwen35/data_common/' + slug + '.jsonl',
                            'sha256': known[slug + '.jsonl']}, 'rows': rows}
            path = args.out / stage / (slug + '.json')
            write_json(path, obj)
            records.append({'path': str(path.relative_to(REPO)), 'sha256': sha256(path),
                            'stage': stage, 'trait': slug, 'rows': len(rows),
                            'max_length': max(len(r[s]['input_ids']) for r in rows for s in ('chosen', 'rejected'))})
            if slug in read_traits:
                examples.append({'stage': stage, 'trait': slug, 'source_row': 0, **rows[0]})
        print(stage, len(panel), 'traits rendered', flush=True)
    write_json(args.out / 'manifest.json', {'schema_version': 1, 'files': records, 'read_traits': read_traits})
    write_json(args.out / 'read_examples.json', examples)
    if 'instruct' in args.stages.split(','):
        prepare_pilot_inputs(args.out / 'instruct', panel, controls, PREFLIGHT / 'pilot_inputs')
    base = checkpoints['base']
    tok = AutoTokenizer.from_pretrained(base['repo'], revision=base['revision'])
    bank = json.loads((HERE / 'activation_bank.json').read_text())
    activation = prepare_activation_inputs(tok, bank, {r['prompt'] for r in raw[panel[0]['slug']]})
    activation['tokenizer_revision'] = base['revision']
    write_json(PREFLIGHT / 'activation_inputs.json', activation)
    print('Saved exact examples:', args.out / 'read_examples.json', flush=True)


if __name__ == '__main__':
    main()

"""Input-format and data-control failures caught before GPU use."""
import copy

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from render import control_rows, render_pair, select_panel, prepare_pilot_inputs, prepare_activation_inputs


@pytest.fixture
def tokenizer():
    backend = Tokenizer(models.WordLevel({
        '<unk>': 0, '<|endoftext|>': 1, '<|user|>': 2, '<|assistant|>': 3,
        'Hello': 4, 'Yes': 5, 'No': 6,
    }, unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>',
                                 eos_token='<|endoftext|>')
    tok.add_special_tokens({'additional_special_tokens': ['<|user|>', '<|assistant|>']})
    tok.chat_template = "{% for message in messages %}{{ '<|' + message['role'] + '|>\\n' + message['content'] + eos_token + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
    return tok


def test_base_has_plain_prompt_and_completion_only_loss(tokenizer):
    r = render_pair(tokenizer, {'prompt': 'Hello', 'chosen': 'Yes', 'rejected': 'No'}, 'base')
    assert r['chosen']['text'] == '<|endoftext|>Hello\n\nYes<|endoftext|>'
    assert r['chosen']['input_ids'] == [1, 4, 5, 1]
    assert r['chosen']['completion_mask'] == [False, False, True, True]


def test_chat_has_one_user_turn_and_no_teacher_constitution(tokenizer):
    r = render_pair(tokenizer, {'prompt': 'Hello', 'chosen': 'Yes', 'rejected': 'No',
                                'constitution': 'SECRET'}, 'instruct')
    assert r['chosen']['text'] == '<|user|>\nHello<|endoftext|>\n<|assistant|>\nYes<|endoftext|>\n'
    assert r['chosen']['input_ids'] == [2, 4, 1, 3, 5, 1]
    assert r['chosen']['completion_mask'] == [False, False, False, False, True, True]


def test_special_token_injection_is_rejected(tokenizer):
    with pytest.raises(ValueError, match='special'):
        render_pair(tokenizer, {'prompt': 'Hello<|endoftext|>', 'chosen': 'Yes', 'rejected': 'No'}, 'base')


def test_long_input_is_rejected_without_truncation(tokenizer):
    with pytest.raises(ValueError, match='length'):
        render_pair(tokenizer, {'prompt': 'Hello Hello', 'chosen': 'Yes', 'rejected': 'No'}, 'base', max_length=4)


def test_empty_or_equal_preference_answers_are_rejected(tokenizer):
    for chosen, rejected in [('', 'No'), ('Yes', 'Yes')]:
        with pytest.raises(ValueError):
            render_pair(tokenizer, {'prompt': 'Hello', 'chosen': chosen, 'rejected': rejected}, 'base')


def test_scrambling_swaps_exactly_half_and_keeps_original_rows():
    rows = [{'prompt': str(i), 'chosen': 'yes', 'rejected': 'no'} for i in range(445)]
    original = copy.deepcopy(rows)
    arm = control_rows(rows, 'scrambled', seed=17)
    assert sum(a['chosen'] == 'no' for a in arm) == 222
    assert [a['prompt'] for a in arm] == [str(i) for i in range(445)]
    assert rows == original
    assert arm == control_rows(rows, 'scrambled', seed=17)


def test_panel_balances_cells_without_inventing_labels():
    entries = [{'trait': w + str(i), 'factor': w, 'keyed': pole}
               for w in ['A', 'B', 'C', 'D', 'E'] for pole in ['+', '-'] for i in range(3)]
    # Distinct positive/negative words, as in the real source.
    for r in entries:
        r['trait'] += 'p' if r['keyed'] == '+' else 'n'
    panel = select_panel(entries)
    assert len(panel) == 20
    assert {r['trait'] for r in panel} == {w + str(i) + p for w in 'ABCDE' for i in [0, 1] for p in ['p', 'n']}
    with pytest.raises(ValueError):
        select_panel(entries[:-2])


def test_control_files_use_the_frozen_data_mapping(tmp_path):
    import json
    source = tmp_path / 'source'
    source.mkdir()
    panel = [{'trait': s, 'slug': s, 'factor': 'A', 'keyed': '+'} for s in ['a', 'b']]
    for s in ['a', 'b']:
        (source / (s + '.json')).write_text(json.dumps({'schema_version': 1, 'trait': {'slug': s},
            'rows': [{'prompt': str(i), 'chosen': {'text': s}, 'rejected': {'text': 'no'}} for i in range(4)]}))
    controls = {'permuted_data_trait': {'a': 'b', 'b': 'a'}, 'scramble_seeds': {'a': 3, 'b': 5},
                'initialization_seeds': {'real': 0, 'scrambled': 0, 'permuted': 0, 'second_seed': 1}, 'order_seed': 0}
    prepare_pilot_inputs(source, panel, controls, tmp_path / 'arms')
    permuted = json.loads((tmp_path / 'arms/permuted/a.json').read_text())
    assert permuted['trait']['slug'] == 'a'
    assert permuted['rows'][0]['chosen']['text'] == 'b'
    assert permuted['control']['data_trait'] == 'b'
    scrambled = json.loads((tmp_path / 'arms/scrambled/a.json').read_text())
    assert sum(r['chosen']['text'] == 'no' for r in scrambled['rows']) == 2


def test_activation_bank_excludes_end_tokens_from_pooling(tokenizer):
    bank = {'schema_version': 1, 'rows': [{'prompt': 'Hello', 'completion': 'Yes'}]}
    result = prepare_activation_inputs(tokenizer, bank, {'unrelated prompt'})
    assert result['rows'][0]['input_ids'] == [1, 4, 5, 1]
    assert result['rows'][0]['completion_mask'] == [False, False, True, False]
    with pytest.raises(ValueError, match='training'):
        prepare_activation_inputs(tokenizer, bank, {'Hello'})

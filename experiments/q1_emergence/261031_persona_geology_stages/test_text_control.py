"""Check contrast direction and row accounting without fetching an encoder."""
import numpy as np
import pytest

from text_control import encode_contrasts


class FixedEncoder:
    # A fixed transform is sufficient to test input/contrast accounting.
    def encode(self, texts, **kwargs):
        table = {'a': [1., 0.], 'b': [0., 1.], 'c': [3., 2.], 'd': [2., 3.]}
        return np.asarray([table[t] for t in texts])


def test_contrasts_keep_trait_and_pair_order():
    corpora = [[{'chosen': 'a', 'rejected': 'b'}, {'chosen': 'c', 'rejected': 'b'}],
               [{'chosen': 'b', 'rejected': 'a'}, {'chosen': 'd', 'rejected': 'a'}]]
    result = encode_contrasts(FixedEncoder(), corpora)
    np.testing.assert_array_equal(result['contrasts'], [[2., 0.], [0., 2.]])
    np.testing.assert_array_equal(result['chosen'][1], [[0., 1.], [2., 3.]])


def test_ragged_corpora_are_rejected():
    with pytest.raises(ValueError, match='pair count'):
        encode_contrasts(FixedEncoder(), [[{'chosen': 'a', 'rejected': 'b'}],
                                          [{'chosen': 'b', 'rejected': 'a'}] * 2])

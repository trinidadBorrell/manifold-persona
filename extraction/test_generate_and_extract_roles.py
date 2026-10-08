import tempfile
import unittest
from pathlib import Path

import numpy as np

from extraction.generate_and_extract_roles import merge_shards


class MergeShardsTest(unittest.TestCase):
    def test_streams_rows_in_record_order(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_path = Path(raw_tmp)
            ckpt_dir = tmp_path / "_ckpt"
            ckpt_dir.mkdir()
            out_dir = tmp_path / "cloud"
            out_dir.mkdir()

            bounds = [(0, 2), (2, 3)]
            expected_avg = np.arange(3 * 2 * 4, dtype=np.float16).reshape(3, 2, 4)
            expected_last = expected_avg + 100
            responses = ["zero", "one", "two"]

            for start, end in bounds:
                np.savez(
                    ckpt_dir / f"shard_{start:06d}_{end:06d}.npz",
                    avg=expected_avg[start:end],
                    last=expected_last[start:end],
                    responses=np.array(responses[start:end], dtype=object),
                    start=start,
                    end=end,
                )

            got_responses, n_layers, hidden = merge_shards(
                bounds, ckpt_dir, out_dir, n_records=3
            )

            got_avg = np.load(out_dir / "prompt_avg.npy", mmap_mode="r")
            got_last = np.load(out_dir / "prompt_last.npy", mmap_mode="r")
            np.testing.assert_array_equal(got_avg, expected_avg)
            np.testing.assert_array_equal(got_last, expected_last)
            self.assertEqual(got_responses, responses)
            self.assertEqual((n_layers, hidden), (2, 4))
            self.assertFalse(list(out_dir.glob("*.tmp.npy")))


if __name__ == "__main__":
    unittest.main()

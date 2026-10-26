import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from diagnostics import inference_diagnostics
from experiments.hybrid_ablation import _one_hot, load_prepared_npz, validate_provenance, feature_hash


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.arrays, self.rows = {}, []
        rng = np.random.default_rng(42)
        for split in ('train', 'val', 'test'):
            self.arrays['X_' + split] = rng.normal(size=(4, 32, 16)).astype('float32')
            self.arrays['y_' + split] = np.arange(4)
            self.arrays['groups_' + split] = np.array([split] * 4)
            for i, label in enumerate(('normal', 'ambient', 'movable_synthetic', 'fixed_synthetic')):
                self.rows.append(dict(split=split, group_id=split, class_name=label,
                                      feature_sha256=feature_hash(self.arrays['X_' + split][i])))

    def validate(self):
        np.savez(self.root / 'data.npz', **self.arrays)
        with (self.root / 'manifest.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader(); writer.writerows(self.rows)
        data = load_prepared_npz(self.root / 'data.npz')
        return validate_provenance(data, self.root / 'manifest.csv')

    def test_valid(self):
        self.assertEqual(self.validate()['feature_binding'], 'verified')

    def test_bad_one_hot(self):
        for y in (np.full((4, 4), .25), np.full((4, 4), np.nan), np.zeros((4, 4)), np.ones((4, 4))):
            with self.subTest(y=y), self.assertRaises(ValueError): _one_hot(y)

    def test_count(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError, 'row count'): self.validate()

    def test_label_order(self):
        self.rows[0]['class_name'], self.rows[1]['class_name'] = self.rows[1]['class_name'], self.rows[0]['class_name']
        with self.assertRaisesRegex(ValueError, 'label order'): self.validate()

    def test_group_leakage(self):
        self.rows[4]['group_id'] = 'train'
        with self.assertRaisesRegex(ValueError, 'leakage'): self.validate()

    def test_feature_binding(self):
        self.arrays['X_train'][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): self.validate()

    def test_duplicate(self):
        self.arrays['X_test'][0] = self.arrays['X_train'][0]
        self.rows[8]['feature_sha256'] = self.rows[0]['feature_sha256']
        with self.assertRaisesRegex(ValueError, 'duplicate'): self.validate()

    def test_embedded_groups(self):
        self.arrays['groups_train'][0] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'groups mismatch'): self.validate()

    def test_shapes(self):
        self.arrays['X_test'] = np.zeros((4, 32, 8))
        with self.assertRaisesRegex(ValueError, 'same feature shape'): self.validate()

    def test_nonfinite_features(self):
        self.arrays['X_train'][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'NaN'): self.validate()

    def test_empty(self):
        self.arrays['X_train'] = np.zeros((0, 32, 16))
        with self.assertRaisesRegex(ValueError, 'nonempty'): self.validate()

    def test_requires_manifest(self):
        with self.assertRaisesRegex(ValueError, 'required'): validate_provenance({}, None)

    def test_legacy_claims(self):
        for row in self.rows: del row['feature_sha256']
        self.assertTrue(self.validate()['feature_binding'].startswith('unverified'))


class DiagnosticsTests(unittest.TestCase):
    def test_uniform_and_certain(self):
        d = inference_diagnostics([[.25] * 4, [1, 0, 0, 0]], [1, 3], list('abcd'))
        self.assertAlmostEqual(d['mean_normalized_entropy'], .5)
        self.assertAlmostEqual(d['mean_top_two_margin'], .5)
        self.assertEqual(d['inference_median_ms'], 2)
        self.assertEqual(d['predicted_class_counts'], dict(a=2, b=0, c=0, d=0))

    def test_invalid(self):
        for p, timing, labels in [([], [], ['a','b']), ([[np.nan, 0]], [1], ['a','b']),
            ([[.8,.8]], [1], ['a','b']), ([[1,0]], [-1], ['a','b']),
            ([[1,0]], [1,2], ['a','b']), ([[1,0]], [1], ['a','a']),
            ([[-.01,1.01]], [1], ['a','b'])]:
            with self.subTest(p=p), self.assertRaises(ValueError): inference_diagnostics(p, timing, labels)

if __name__ == '__main__': unittest.main()

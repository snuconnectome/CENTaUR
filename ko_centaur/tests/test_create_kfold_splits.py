#!/usr/bin/env python3
"""
Test suite for k-fold split generator
Following TDD (Test-Driven Development) practice

Tests written BEFORE implementation to define expected behavior.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from collections import Counter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestKFoldSplitGenerator(unittest.TestCase):
    """Test k-fold cross-validation split generation"""

    def setUp(self):
        """Create temporary test data"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_path = os.path.join(self.temp_dir, "test_data.jsonl")

        # Create sample data (100 samples: 45 class 0, 55 class 1)
        self.sample_data = []
        for i in range(45):
            self.sample_data.append({
                "text": f"Sample {i} for class 0",
                "choice": 0
            })
        for i in range(55):
            self.sample_data.append({
                "text": f"Sample {i+45} for class 1",
                "choice": 1
            })

        # Write test data
        with open(self.test_data_path, 'w') as f:
            for item in self.sample_data:
                f.write(json.dumps(item) + '\n')

    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_import_module(self):
        """Test 1: Module can be imported"""
        try:
            import create_kfold_splits
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import module: {e}")

    def test_create_stratified_kfold_function_exists(self):
        """Test 2: create_stratified_kfold function exists"""
        import create_kfold_splits
        self.assertTrue(hasattr(create_kfold_splits, 'create_stratified_kfold'))

    def test_kfold_returns_correct_number_of_folds(self):
        """Test 3: Creates exactly n_splits folds"""
        import create_kfold_splits

        folds = create_kfold_splits.create_stratified_kfold(
            data_path=self.test_data_path,
            n_splits=10,
            seed=42
        )

        self.assertEqual(len(folds), 10, "Should create exactly 10 folds")

    def test_folds_contain_train_test_splits(self):
        """Test 4: Each fold has train and test indices"""
        import create_kfold_splits

        folds = create_kfold_splits.create_stratified_kfold(
            data_path=self.test_data_path,
            n_splits=10,
            seed=42
        )

        for i, fold in enumerate(folds):
            self.assertIn('train', fold, f"Fold {i} missing 'train' key")
            self.assertIn('test', fold, f"Fold {i} missing 'test' key")

    def test_train_test_no_overlap(self):
        """Test 5: Train and test indices do not overlap"""
        import create_kfold_splits

        folds = create_kfold_splits.create_stratified_kfold(
            data_path=self.test_data_path,
            n_splits=10,
            seed=42
        )

        for i, fold in enumerate(folds):
            train_set = set(fold['train'])
            test_set = set(fold['test'])
            overlap = train_set & test_set
            self.assertEqual(len(overlap), 0,
                           f"Fold {i}: train and test have {len(overlap)} overlapping indices")

    def test_all_samples_used_exactly_once_in_test(self):
        """Test 6: Every sample appears exactly once as test data across folds"""
        import create_kfold_splits

        folds = create_kfold_splits.create_stratified_kfold(
            data_path=self.test_data_path,
            n_splits=10,
            seed=42
        )

        all_test_indices = []
        for fold in folds:
            all_test_indices.extend(fold['test'])

        # Check all indices used
        expected_indices = set(range(len(self.sample_data)))
        actual_indices = set(all_test_indices)
        self.assertEqual(actual_indices, expected_indices,
                        "Not all samples appear as test data")

        # Check no duplicates
        counter = Counter(all_test_indices)
        duplicates = {idx: count for idx, count in counter.items() if count > 1}
        self.assertEqual(len(duplicates), 0,
                        f"Some samples appear multiple times as test: {duplicates}")

    def test_class_balance_maintained_in_folds(self):
        """Test 7: Class balance maintained in each fold (stratified sampling)"""
        import create_kfold_splits

        folds = create_kfold_splits.create_stratified_kfold(
            data_path=self.test_data_path,
            n_splits=10,
            seed=42
        )

        # Overall class distribution: 45% class 0, 55% class 1
        for i, fold in enumerate(folds):
            test_samples = [self.sample_data[idx] for idx in fold['test']]
            test_choices = [s['choice'] for s in test_samples]

            class_0_count = sum(1 for c in test_choices if c == 0)
            class_1_count = sum(1 for c in test_choices if c == 1)
            total = len(test_choices)

            class_0_pct = class_0_count / total if total > 0 else 0

            # Allow ±10% deviation from target 45%
            self.assertGreaterEqual(class_0_pct, 0.35,
                                  f"Fold {i}: class 0 proportion {class_0_pct:.2%} too low")
            self.assertLessEqual(class_0_pct, 0.55,
                               f"Fold {i}: class 0 proportion {class_0_pct:.2%} too high")

    def test_fold_sizes_approximately_equal(self):
        """Test 8: Test folds are approximately equal in size"""
        import create_kfold_splits

        folds = create_kfold_splits.create_stratified_kfold(
            data_path=self.test_data_path,
            n_splits=10,
            seed=42
        )

        test_sizes = [len(fold['test']) for fold in folds]
        expected_size = len(self.sample_data) / 10  # 10 samples per fold

        for i, size in enumerate(test_sizes):
            # Allow ±1 sample difference
            self.assertAlmostEqual(size, expected_size, delta=1,
                                 msg=f"Fold {i} test size {size} differs too much from expected {expected_size}")

    def test_reproducibility_with_same_seed(self):
        """Test 9: Same seed produces identical splits"""
        import create_kfold_splits

        folds1 = create_kfold_splits.create_stratified_kfold(
            data_path=self.test_data_path,
            n_splits=10,
            seed=42
        )

        folds2 = create_kfold_splits.create_stratified_kfold(
            data_path=self.test_data_path,
            n_splits=10,
            seed=42
        )

        for i, (fold1, fold2) in enumerate(zip(folds1, folds2)):
            self.assertEqual(fold1['train'], fold2['train'],
                           f"Fold {i} train indices differ between runs")
            self.assertEqual(fold1['test'], fold2['test'],
                           f"Fold {i} test indices differ between runs")

    def test_different_seeds_produce_different_splits(self):
        """Test 10: Different seeds produce different splits"""
        import create_kfold_splits

        folds1 = create_kfold_splits.create_stratified_kfold(
            data_path=self.test_data_path,
            n_splits=10,
            seed=42
        )

        folds2 = create_kfold_splits.create_stratified_kfold(
            data_path=self.test_data_path,
            n_splits=10,
            seed=123
        )

        # At least one fold should be different
        any_different = False
        for fold1, fold2 in zip(folds1, folds2):
            if fold1['test'] != fold2['test']:
                any_different = True
                break

        self.assertTrue(any_different, "Different seeds should produce different splits")

    def test_save_folds_function_exists(self):
        """Test 11: save_folds function exists"""
        import create_kfold_splits
        self.assertTrue(hasattr(create_kfold_splits, 'save_folds'))

    def test_save_folds_creates_files(self):
        """Test 12: save_folds creates JSON files for each fold"""
        import create_kfold_splits

        folds = create_kfold_splits.create_stratified_kfold(
            data_path=self.test_data_path,
            n_splits=10,
            seed=42
        )

        output_dir = os.path.join(self.temp_dir, "folds")
        create_kfold_splits.save_folds(folds, output_dir)

        # Check all fold files created
        for i in range(10):
            fold_file = os.path.join(output_dir, f"fold_{i}.json")
            self.assertTrue(os.path.exists(fold_file),
                          f"Fold file {fold_file} not created")

    def test_saved_folds_loadable(self):
        """Test 13: Saved folds can be loaded back"""
        import create_kfold_splits

        folds = create_kfold_splits.create_stratified_kfold(
            data_path=self.test_data_path,
            n_splits=10,
            seed=42
        )

        output_dir = os.path.join(self.temp_dir, "folds")
        create_kfold_splits.save_folds(folds, output_dir)

        # Load and verify
        for i in range(10):
            fold_file = os.path.join(output_dir, f"fold_{i}.json")
            with open(fold_file, 'r') as f:
                loaded_fold = json.load(f)

            self.assertIn('train', loaded_fold)
            self.assertIn('test', loaded_fold)
            self.assertEqual(loaded_fold['train'], folds[i]['train'])
            self.assertEqual(loaded_fold['test'], folds[i]['test'])

    def test_validate_folds_function_exists(self):
        """Test 14: validate_folds function exists"""
        import create_kfold_splits
        self.assertTrue(hasattr(create_kfold_splits, 'validate_folds'))

    def test_validate_folds_returns_metrics(self):
        """Test 15: validate_folds returns metrics dictionary"""
        import create_kfold_splits

        folds = create_kfold_splits.create_stratified_kfold(
            data_path=self.test_data_path,
            n_splits=10,
            seed=42
        )

        metrics = create_kfold_splits.validate_folds(
            folds,
            self.sample_data
        )

        # Check required metrics exist
        self.assertIsInstance(metrics, dict)
        self.assertIn('n_splits', metrics)
        self.assertIn('total_samples', metrics)
        self.assertIn('fold_sizes', metrics)
        self.assertIn('class_balance', metrics)


class TestKFoldSplitGeneratorIntegration(unittest.TestCase):
    """Integration tests for k-fold split generator with real data"""

    def setUp(self):
        """Set up paths for real data"""
        self.project_root = Path(__file__).parent.parent
        self.data_path = self.project_root / "data" / "choices13k_1000.jsonl"
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)

    @unittest.skipUnless(
        Path(__file__).parent.parent.joinpath("data/choices13k_1000.jsonl").exists(),
        "choices13k_1000.jsonl not found"
    )
    def test_integration_with_real_data(self):
        """Test 16: Integration test with real choices13k_1000 data"""
        import create_kfold_splits

        # Create folds
        folds = create_kfold_splits.create_stratified_kfold(
            data_path=str(self.data_path),
            n_splits=10,
            seed=42
        )

        # Validate
        self.assertEqual(len(folds), 10)

        # Check each fold
        for i, fold in enumerate(folds):
            # Approximately 100 samples per fold
            test_size = len(fold['test'])
            self.assertGreaterEqual(test_size, 90)
            self.assertLessEqual(test_size, 110)

            # Train size should be ~900
            train_size = len(fold['train'])
            self.assertGreaterEqual(train_size, 890)
            self.assertLessEqual(train_size, 910)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()

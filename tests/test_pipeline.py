"""
Integration Tests for Full Pipeline
===================================
Tests the complete pipeline from data preparation to model training.
"""

import unittest
import subprocess
import sys
from pathlib import Path
import pandas as pd
import joblib


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for complete pipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.project_root = Path(__file__).parent.parent
        cls.pipeline_script = cls.project_root / 'scripts' / 'run_pipeline.py'
        cls.data_file = cls.project_root / 'data' / 'processed' / 'covid19_prepared_data.csv'
        cls.model_file = cls.project_root / 'models' / 'trained' / 'best_covid_warning_model.pkl'
    
    def test_pipeline_script_exists(self):
        """Test that pipeline script exists"""
        self.assertTrue(self.pipeline_script.exists(),
                       f"Pipeline script should exist at {self.pipeline_script}")
    
    def test_data_preparation_module_exists(self):
        """Test that data preparation module exists"""
        data_module = self.project_root / 'src' / 'data' / 'prepare_data.py'
        self.assertTrue(data_module.exists(),
                       f"Data preparation module should exist at {data_module}")
    
    def test_model_training_module_exists(self):
        """Test that model training module exists"""
        model_module = self.project_root / 'src' / 'models' / 'train_model.py'
        self.assertTrue(model_module.exists(),
                       f"Model training module should exist at {model_module}")
    
    def test_end_to_end_pipeline(self):
        """Test that complete pipeline runs successfully"""
        # This test actually runs the pipeline - can be slow
        result = subprocess.run(
            [sys.executable, str(self.pipeline_script)],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )
        
        self.assertEqual(result.returncode, 0,
                        f"Pipeline should complete successfully.\nStderr: {result.stderr}")
    
    def test_pipeline_output_files(self):
        """Test that pipeline creates all expected output files"""
        expected_files = [
            self.data_file,
            self.model_file,
            self.project_root / 'models' / 'trained' / 'model_metadata.pkl',
            self.project_root / 'models' / 'trained' / 'per_class_performance.csv'
        ]
        
        for file_path in expected_files:
            self.assertTrue(file_path.exists(),
                          f"Expected output file should exist: {file_path}")
    
    def test_data_model_compatibility(self):
        """Test that model is compatible with prepared data"""
        if self.data_file.exists() and self.model_file.exists():
            # Load data
            df = pd.read_csv(self.data_file)
            X = df.drop('Warning_Level', axis=1, errors='ignore')
            
            # Load model
            artifact = joblib.load(self.model_file)
            model = artifact['model']
            expected_features = artifact['feature_names']
            
            # Check feature compatibility
            self.assertEqual(len(expected_features), len(X.columns),
                           "Model expects same number of features as data")
            
            # Check feature names match
            self.assertEqual(set(expected_features), set(X.columns),
                           "Model features should match data columns")
    
    def test_model_predicts_on_prepared_data(self):
        """Test that model can make predictions on prepared data"""
        if self.data_file.exists() and self.model_file.exists():
            # Load data
            df = pd.read_csv(self.data_file)
            X = df.drop('Warning_Level', axis=1, errors='ignore').head(10)
            
            # Load model
            artifact = joblib.load(self.model_file)
            model = artifact['model']
            
            try:
                predictions = model.predict(X)
                self.assertEqual(len(predictions), 10,
                               "Should predict for all 10 samples")
            except Exception as e:
                self.fail(f"Model should predict on prepared data: {e}")


class TestPipelineComponents(unittest.TestCase):
    """Test individual pipeline components"""
    
    def test_data_preparation_standalone(self):
        """Test data preparation can run standalone"""
        result = subprocess.run(
            [sys.executable, '-m', 'src.data.prepare_data'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        # Should complete without error
        self.assertEqual(result.returncode, 0,
                        f"Data preparation should run standalone.\nStderr: {result.stderr}")
    
    def test_model_training_with_data(self):
        """Test model training can run when data exists"""
        # Ensure data exists first
        subprocess.run(
            [sys.executable, '-m', 'src.data.prepare_data'],
            capture_output=True,
            cwd=Path(__file__).parent.parent
        )
        
        # Run training
        result = subprocess.run(
            [sys.executable, '-m', 'src.models.train_model'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        self.assertEqual(result.returncode, 0,
                        f"Model training should run successfully.\nStderr: {result.stderr}")


class TestTestDataCompatibility(unittest.TestCase):
    """Test that test data files are compatible with model"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.project_root = Path(__file__).parent.parent
        cls.test_data_dir = cls.project_root / 'tests' / 'test_data'
        cls.model_file = cls.project_root / 'models' / 'trained' / 'best_covid_warning_model.pkl'
    
    def test_test_data_directory_exists(self):
        """Test that test data directory exists"""
        self.assertTrue(self.test_data_dir.exists(),
                       f"Test data directory should exist at {self.test_data_dir}")
    
    def test_test_csv_files_exist(self):
        """Test that test CSV files exist"""
        test_files = list(self.test_data_dir.glob('*.csv'))
        self.assertGreater(len(test_files), 0,
                         "Should have at least one test CSV file")
    
    def test_test_files_loadable(self):
        """Test that test CSV files can be loaded"""
        test_files = list(self.test_data_dir.glob('*.csv'))
        
        for test_file in test_files:
            try:
                df = pd.read_csv(test_file)
                self.assertIsInstance(df, pd.DataFrame,
                                    f"{test_file.name} should load as DataFrame")
            except Exception as e:
                self.fail(f"Test file {test_file.name} should be loadable: {e}")
    
    def test_test_data_model_compatibility(self):
        """Test that test data has same features as model expects"""
        if not self.model_file.exists():
            self.skipTest("Model not found, skipping compatibility test")
        
        # Load model
        artifact = joblib.load(self.model_file)
        expected_features = set(artifact['feature_names'])
        
        # Check each test file
        test_files = list(self.test_data_dir.glob('*.csv'))
        
        for test_file in test_files:
            df = pd.read_csv(test_file)
            test_features = set(df.columns)
            
            # Test data might have extra columns like 'Expected_Warning', that's OK
            # But it must have all required features
            missing_features = expected_features - test_features
            
            self.assertEqual(len(missing_features), 0,
                           f"{test_file.name} missing features: {missing_features}")


if __name__ == '__main__':
    unittest.main()

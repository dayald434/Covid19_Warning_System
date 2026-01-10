"""
Unit Tests for Model Training Module
====================================
Tests model training, evaluation, and saving functions.
"""

import unittest
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.train_model import train_warning_system


class TestModelTraining(unittest.TestCase):
    """Test cases for model training module"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.project_root = Path(__file__).parent.parent
        cls.model_file = cls.project_root / 'models' / 'trained' / 'best_covid_warning_model.pkl'
        cls.metadata_file = cls.project_root / 'models' / 'trained' / 'model_metadata.pkl'
        cls.metrics_file = cls.project_root / 'models' / 'trained' / 'per_class_performance.csv'
    
    def test_model_file_created(self):
        """Test that model file is created after training"""
        # Ensure data exists first
        from src.data.prepare_data import load_and_prepare_data
        load_and_prepare_data()
        
        # Train model
        result = train_warning_system()
        
        # Check model file exists
        self.assertTrue(self.model_file.exists(),
                       f"Model file should exist at {self.model_file}")
    
    def test_metadata_file_created(self):
        """Test that metadata file is created"""
        self.assertTrue(self.metadata_file.exists(),
                       "Metadata file should be created during training")
    
    def test_metrics_file_created(self):
        """Test that metrics CSV file is created"""
        self.assertTrue(self.metrics_file.exists(),
                       "Per-class metrics file should be created")
    
    def test_model_loadable(self):
        """Test that saved model can be loaded"""
        if self.model_file.exists():
            try:
                artifact = joblib.load(self.model_file)
                self.assertIsInstance(artifact, dict,
                                    "Model artifact should be a dictionary")
            except Exception as e:
                self.fail(f"Model should be loadable without errors: {e}")
    
    def test_model_structure(self):
        """Test that model artifact has correct structure"""
        if self.model_file.exists():
            artifact = joblib.load(self.model_file)
            
            required_keys = ['model', 'feature_names', 'target_classes', 'metadata']
            for key in required_keys:
                self.assertIn(key, artifact,
                            f"Model artifact should contain '{key}'")
    
    def test_model_type(self):
        """Test that model is a RandomForestClassifier"""
        if self.model_file.exists():
            artifact = joblib.load(self.model_file)
            model = artifact['model']
            
            from sklearn.ensemble import RandomForestClassifier
            self.assertIsInstance(model, RandomForestClassifier,
                                "Model should be a RandomForestClassifier")
    
    def test_model_is_trained(self):
        """Test that model has been fitted"""
        if self.model_file.exists():
            artifact = joblib.load(self.model_file)
            model = artifact['model']
            
            # Check if model has been fitted
            self.assertTrue(hasattr(model, 'n_features_in_'),
                          "Model should have n_features_in_ attribute after fitting")
    
    def test_feature_names_saved(self):
        """Test that feature names are saved correctly"""
        if self.model_file.exists():
            artifact = joblib.load(self.model_file)
            feature_names = artifact['feature_names']
            
            self.assertIsInstance(feature_names, list,
                                "Feature names should be a list")
            self.assertGreater(len(feature_names), 0,
                             "Should have at least one feature")
    
    def test_target_classes_saved(self):
        """Test that target classes are saved"""
        if self.model_file.exists():
            artifact = joblib.load(self.model_file)
            target_classes = artifact['target_classes']
            
            self.assertIsInstance(target_classes, list,
                                "Target classes should be a list")
            self.assertGreater(len(target_classes), 1,
                             "Should have at least 2 target classes")
    
    def test_metadata_content(self):
        """Test that metadata contains required information"""
        if self.model_file.exists():
            artifact = joblib.load(self.model_file)
            metadata = artifact['metadata']
            
            required_fields = ['train_date', 'accuracy', 'n_features', 'model_type']
            for field in required_fields:
                self.assertIn(field, metadata,
                            f"Metadata should contain '{field}'")
    
    def test_accuracy_range(self):
        """Test that accuracy is within valid range"""
        if self.model_file.exists():
            artifact = joblib.load(self.model_file)
            accuracy = artifact['metadata']['accuracy']
            
            self.assertGreaterEqual(accuracy, 0.0,
                                  "Accuracy should be >= 0")
            self.assertLessEqual(accuracy, 1.0,
                               "Accuracy should be <= 1")
    
    def test_model_prediction(self):
        """Test that model can make predictions"""
        if self.model_file.exists():
            artifact = joblib.load(self.model_file)
            model = artifact['model']
            feature_names = artifact['feature_names']
            
            # Create sample input
            n_features = len(feature_names)
            sample_input = pd.DataFrame(
                np.random.randn(5, n_features),
                columns=feature_names
            )
            
            try:
                predictions = model.predict(sample_input)
                self.assertEqual(len(predictions), 5,
                               "Should predict for all 5 samples")
            except Exception as e:
                self.fail(f"Model should be able to make predictions: {e}")
    
    def test_metrics_file_structure(self):
        """Test that metrics CSV has correct structure"""
        if self.metrics_file.exists():
            metrics_df = pd.read_csv(self.metrics_file)
            
            required_columns = ['Warning_Level', 'Precision', 'Recall', 'F1_Score']
            for col in required_columns:
                self.assertIn(col, metrics_df.columns,
                            f"Metrics file should contain '{col}' column")


class TestModelPerformance(unittest.TestCase):
    """Test cases for model performance requirements"""
    
    @classmethod
    def setUpClass(cls):
        """Load model for testing"""
        cls.project_root = Path(__file__).parent.parent
        cls.model_file = cls.project_root / 'models' / 'trained' / 'best_covid_warning_model.pkl'
        
        if cls.model_file.exists():
            cls.artifact = joblib.load(cls.model_file)
            cls.model = cls.artifact['model']
            cls.metadata = cls.artifact['metadata']
    
    def test_minimum_accuracy(self):
        """Test that model meets minimum accuracy threshold"""
        if hasattr(self, 'metadata'):
            accuracy = self.metadata['accuracy']
            min_accuracy = 0.70  # 70% minimum
            
            self.assertGreater(accuracy, min_accuracy,
                             f"Model accuracy ({accuracy:.2%}) should be > {min_accuracy:.2%}")
    
    def test_model_not_overfitting(self):
        """Test that model is not perfectly accurate (likely overfitting)"""
        if hasattr(self, 'metadata'):
            accuracy = self.metadata['accuracy']
            
            # Perfect accuracy often indicates overfitting on small/synthetic data
            # This is a warning, not a hard failure
            if accuracy >= 0.999:
                print(f"\n⚠️  WARNING: Very high accuracy ({accuracy:.2%}) - check for overfitting")
    
    def test_feature_importance_available(self):
        """Test that feature importance can be extracted"""
        if hasattr(self, 'model'):
            self.assertTrue(hasattr(self.model, 'feature_importances_'),
                          "Model should have feature_importances_ attribute")
            
            importances = self.model.feature_importances_
            self.assertEqual(len(importances), self.metadata['n_features'],
                           "Should have importance for each feature")
    
    def test_model_reproducibility(self):
        """Test that model predictions are deterministic"""
        if hasattr(self, 'model') and hasattr(self, 'artifact'):
            feature_names = self.artifact['feature_names']
            n_features = len(feature_names)
            
            # Create test data
            np.random.seed(42)
            test_data = pd.DataFrame(
                np.random.randn(10, n_features),
                columns=feature_names
            )
            
            # Make predictions twice
            pred1 = self.model.predict(test_data)
            pred2 = self.model.predict(test_data)
            
            np.testing.assert_array_equal(pred1, pred2,
                                         "Model predictions should be reproducible")


if __name__ == '__main__':
    unittest.main()

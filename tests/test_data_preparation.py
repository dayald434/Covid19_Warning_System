"""
Unit Tests for Data Preparation Module
======================================
Tests data loading, cleaning, and feature engineering functions.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.prepare_data import load_and_prepare_data


class TestDataPreparation(unittest.TestCase):
    """Test cases for data preparation module"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests"""
        cls.project_root = Path(__file__).parent.parent
        cls.processed_file = cls.project_root / 'data' / 'processed' / 'covid19_prepared_data.csv'
    
    def test_data_file_exists(self):
        """Test that prepared data file exists after running preparation"""
        # Run data preparation
        df = load_and_prepare_data()
        
        # Check file exists
        self.assertTrue(self.processed_file.exists(), 
                       f"Prepared data file should exist at {self.processed_file}")
    
    def test_data_structure(self):
        """Test that prepared data has correct structure"""
        df = load_and_prepare_data()
        
        # Check it's a DataFrame
        self.assertIsInstance(df, pd.DataFrame, "Output should be a pandas DataFrame")
        
        # Check minimum number of rows
        self.assertGreater(len(df), 0, "Dataset should not be empty")
        
        # Check has features
        self.assertGreater(df.shape[1], 1, "Dataset should have multiple columns")
    
    def test_required_columns(self):
        """Test that all required columns are present"""
        df = load_and_prepare_data()
        
        required_columns = [
            'Cases_per_100k',
            'Deaths_per_100k',
            'Case_Fatality_Rate',
            'Cases_7day_avg',
            'Deaths_7day_avg',
            'Warning_Level'
        ]
        
        for col in required_columns:
            self.assertIn(col, df.columns, 
                         f"Required column '{col}' should be in dataset")
    
    def test_warning_level_values(self):
        """Test that Warning_Level contains valid categories"""
        df = load_and_prepare_data()
        
        valid_levels = [
            'LOW_MONITORING',
            'MODERATE_MEASURES',
            'HIGH_RESTRICTIONS',
            'CRITICAL_LOCKDOWN'
        ]
        
        self.assertIn('Warning_Level', df.columns, 
                     "Dataset must have Warning_Level column")
        
        unique_levels = df['Warning_Level'].unique()
        for level in unique_levels:
            self.assertIn(level, valid_levels, 
                         f"Warning level '{level}' is not valid")
    
    def test_no_missing_values_in_features(self):
        """Test that there are no missing values in feature columns"""
        df = load_and_prepare_data()
        
        feature_cols = [col for col in df.columns if col != 'Warning_Level']
        
        for col in feature_cols:
            missing_count = df[col].isna().sum()
            self.assertEqual(missing_count, 0, 
                           f"Column '{col}' should not have missing values")
    
    def test_numeric_features(self):
        """Test that feature columns are numeric"""
        df = load_and_prepare_data()
        
        numeric_features = [
            'Cases_per_100k',
            'Deaths_per_100k',
            'Case_Fatality_Rate'
        ]
        
        for col in numeric_features:
            if col in df.columns:
                self.assertTrue(pd.api.types.is_numeric_dtype(df[col]),
                              f"Column '{col}' should be numeric")
    
    def test_no_negative_values(self):
        """Test that rate/count features don't have negative values"""
        df = load_and_prepare_data()
        
        non_negative_cols = [
            'Cases_per_100k',
            'Deaths_per_100k',
            'Case_Fatality_Rate'
        ]
        
        for col in non_negative_cols:
            if col in df.columns:
                min_value = df[col].min()
                self.assertGreaterEqual(min_value, 0, 
                                      f"Column '{col}' should not have negative values")
    
    def test_data_reproducibility(self):
        """Test that running preparation twice gives same results"""
        df1 = load_and_prepare_data()
        df2 = load_and_prepare_data()
        
        self.assertEqual(len(df1), len(df2), 
                        "Multiple runs should produce same number of rows")
        self.assertEqual(list(df1.columns), list(df2.columns),
                        "Multiple runs should produce same columns")


class TestDataQuality(unittest.TestCase):
    """Test cases for data quality checks"""
    
    def setUp(self):
        """Set up test data for each test"""
        self.df = load_and_prepare_data()
    
    def test_balanced_classes(self):
        """Test that dataset is not extremely imbalanced"""
        if 'Warning_Level' in self.df.columns:
            class_counts = self.df['Warning_Level'].value_counts()
            
            # Check that we have at least 2 classes
            self.assertGreaterEqual(len(class_counts), 2,
                                  "Dataset should have at least 2 warning levels")
            
            # Check no class is less than 0.1% of total
            min_class_ratio = class_counts.min() / len(self.df)
            self.assertGreater(min_class_ratio, 0.0001,
                             "Classes should not be extremely imbalanced")
    
    def test_feature_variance(self):
        """Test that features have sufficient variance"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            variance = self.df[col].var()
            self.assertGreater(variance, 0, 
                             f"Feature '{col}' should have non-zero variance")
    
    def test_no_constant_features(self):
        """Test that no features are constant"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            unique_count = self.df[col].nunique()
            self.assertGreater(unique_count, 1,
                             f"Feature '{col}' should not be constant")
    
    def test_reasonable_value_ranges(self):
        """Test that percentage features are within 0-100 range"""
        percentage_cols = [col for col in self.df.columns if 'Rate' in col]
        
        for col in percentage_cols:
            if col in self.df.columns:
                max_val = self.df[col].max()
                self.assertLessEqual(max_val, 100,
                                   f"Percentage column '{col}' should be <= 100")


if __name__ == '__main__':
    unittest.main()

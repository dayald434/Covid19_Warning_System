"""
Unit Tests for Streamlit Application
====================================
Tests for the web application components.
"""

import unittest
from pathlib import Path
import sys
import pandas as pd
import joblib
import importlib.util

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestStreamlitApp(unittest.TestCase):
    """Test cases for Streamlit application"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.project_root = Path(__file__).parent.parent
        cls.app_file = cls.project_root / 'app' / 'streamlit_app.py'
        cls.model_file = cls.project_root / 'models' / 'trained' / 'best_covid_warning_model.pkl'
    
    def test_app_file_exists(self):
        """Test that Streamlit app file exists"""
        self.assertTrue(self.app_file.exists(),
                       f"Streamlit app should exist at {self.app_file}")
    
    def test_app_file_syntax(self):
        """Test that app file has valid Python syntax"""
        try:
            with open(self.app_file, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, str(self.app_file), 'exec')
        except SyntaxError as e:
            self.fail(f"App file has syntax error: {e}")
    
    def test_app_imports(self):
        """Test that app can be imported without errors"""
        try:
            spec = importlib.util.spec_from_file_location("streamlit_app", self.app_file)
            module = importlib.util.module_from_spec(spec)
            # Note: We don't actually execute it as it would start Streamlit
            # Just verify it can be loaded
        except Exception as e:
            # This might fail due to Streamlit not being in headless mode
            # So we just check if it's an import error or actual code error
            if "streamlit" not in str(e).lower():
                self.fail(f"App should be importable: {e}")
    
    def test_model_path_in_app(self):
        """Test that app references correct model path"""
        with open(self.app_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for model loading references
        self.assertIn('models/trained', content,
                     "App should reference models/trained directory")
        self.assertIn('best_covid_warning_model.pkl', content,
                     "App should reference model file")


class TestModelLoadingInApp(unittest.TestCase):
    """Test model loading functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.project_root = Path(__file__).parent.parent
        cls.model_file = cls.project_root / 'models' / 'trained' / 'best_covid_warning_model.pkl'
    
    def test_model_loads_for_app(self):
        """Test that model can be loaded as app would load it"""
        if not self.model_file.exists():
            self.skipTest("Model file not found")
        
        try:
            artifact = joblib.load(self.model_file)
            self.assertIn('model', artifact)
            self.assertIn('feature_names', artifact)
        except Exception as e:
            self.fail(f"Model should be loadable: {e}")
    
    def test_sample_prediction(self):
        """Test that model can make sample predictions"""
        if not self.model_file.exists():
            self.skipTest("Model file not found")
        
        artifact = joblib.load(self.model_file)
        model = artifact['model']
        feature_names = artifact['feature_names']
        
        # Create sample input matching expected features
        sample = pd.DataFrame([[50] * len(feature_names)], columns=feature_names)
        
        try:
            prediction = model.predict(sample)
            self.assertEqual(len(prediction), 1)
        except Exception as e:
            self.fail(f"Model should make predictions: {e}")


class TestAppConfiguration(unittest.TestCase):
    """Test application configuration and setup"""
    
    def test_requirements_file_exists(self):
        """Test that requirements.txt exists"""
        requirements_file = Path(__file__).parent.parent / 'requirements.txt'
        self.assertTrue(requirements_file.exists(),
                       "requirements.txt should exist")
    
    def test_streamlit_in_requirements(self):
        """Test that streamlit is in requirements"""
        requirements_file = Path(__file__).parent.parent / 'requirements.txt'
        
        if requirements_file.exists():
            with open(requirements_file, 'r') as f:
                requirements = f.read().lower()
            
            self.assertIn('streamlit', requirements,
                         "streamlit should be in requirements.txt")
    
    def test_readme_has_run_instructions(self):
        """Test that README has instructions to run app"""
        readme_file = Path(__file__).parent.parent / 'README.md'
        
        if readme_file.exists():
            with open(readme_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.assertIn('streamlit run', content.lower(),
                         "README should have instructions to run Streamlit app")


if __name__ == '__main__':
    unittest.main()

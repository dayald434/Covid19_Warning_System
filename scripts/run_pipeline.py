"""
COVID-19 Warning System - Main Pipeline Runner
==============================================
Execute the complete pipeline: data preparation → model training → predictions

Usage:
    python scripts/run_pipeline.py              # Run full pipeline
    python scripts/run_pipeline.py --prepare    # Data preparation only
    python scripts/run_pipeline.py --train      # Training only
    python scripts/run_pipeline.py --predict    # Make predictions (requires trained model)
"""

import os
import sys
import subprocess
from pathlib import Path

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print("\n" + "=" * 80)
    print(f"{description}")
    print("=" * 80)
    
    script_path = Path(__file__).parent.parent / 'src' / script_name
    
    if not script_path.exists():
        print(f"❌ ERROR: {script_name} not found at {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path(__file__).parent.parent,
            capture_output=False
        )
        
        if result.returncode == 0:
            print(f"\n✅ {description} completed successfully!")
            return True
        else:
            print(f"\n❌ {description} failed with return code {result.returncode}")
            return False
    
    except Exception as e:
        print(f"❌ ERROR running {script_name}: {str(e)}")
        return False

def main():
    """Main pipeline orchestrator"""
    
    print("=" * 80)
    print("COVID-19 EARLY WARNING SYSTEM - MAIN PIPELINE")
    print("=" * 80)
    print("\nProject Goal: Predict required public health actions 7 days in advance")
    print("without predicting future outbreaks\n")
    
    # Check command line arguments
    full_pipeline = True
    prepare_only = '--prepare' in sys.argv
    train_only = '--train' in sys.argv
    predict_only = '--predict' in sys.argv
    
    if prepare_only or train_only or predict_only:
        full_pipeline = False
    
    # ========================================================================
    # STEP 1: Data Preparation
    # ========================================================================
    if prepare_only or full_pipeline:
        success = run_script(
            'data/prepare_data.py',
            '[STEP 1] DATA PREPARATION - Cleaning & Feature Engineering'
        )
        if not success and full_pipeline:
            print("\n⚠️  Data preparation failed. Cannot continue.")
            return False
    
    # ========================================================================
    # STEP 2: Model Training
    # ========================================================================
    if train_only or full_pipeline:
        success = run_script(
            'models/train_model.py',
            '[STEP 2] MODEL TRAINING - Building & Evaluating Warning System'
        )
        if not success and full_pipeline:
            print("\n⚠️  Model training failed.")
            return False
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    if full_pipeline:
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE! ✅")
        print("=" * 80)
        
        project_root = Path(__file__).parent.parent
        print("\nGenerated files:")
        
        output_files = [
            ('data/processed/covid19_prepared_data.csv', 'Cleaned dataset with 40+ features'),
            ('models/trained/best_covid_warning_model.pkl', 'Production-ready model'),
            ('models/trained/model_metadata.pkl', 'Model documentation & metrics'),
            ('models/trained/per_class_performance.csv', 'Per-class metrics'),
        ]
        
        for filepath_str, description in output_files:
            filepath = project_root / filepath_str
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"  ✓ {filepath_str} ({size_mb:.2f} MB) - {description}")
            else:
                print(f"  ✗ {filepath_str} - NOT FOUND")
        
        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("""
1. Check model performance:
   - models/trained/per_class_performance.csv
   
2. Load and use the model:
   - See README.md for examples
   - Use models/trained/best_covid_warning_model.pkl for predictions
   
3. Run the web interface:
   - streamlit run app/streamlit_app.py
   
4. Test with sample data:
   - Upload CSV files from tests/test_data/
   
5. Deploy to production:
   - Set up data pipeline for real-time data
   - Create API endpoint for predictions
   - Monitor model performance
   - Retrain monthly with new data

For detailed documentation, see README.md in project root.
        """)
    
    return True

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)

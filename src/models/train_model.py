"""
COVID-19 Warning System Model Training
======================================
Trains a Random Forest classifier to predict warning levels 7 days in advance.

Generates:
- models/trained/best_covid_warning_model.pkl
- models/trained/model_metadata.pkl
- models/trained/per_class_performance.csv
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score

def train_warning_system():
    """Train the COVID-19 Warning System model"""
    
    print("\n" + "="*80)
    print("COVID-19 WARNING SYSTEM - MODEL TRAINING")
    print("="*80)
    
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    data_file = project_root / 'data' / 'processed' / 'covid19_prepared_data.csv'
    models_dir = project_root / 'models' / 'trained'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prepared data
    print(f"\n[1/5] Loading prepared data...")
    if not data_file.exists():
        print(f"❌ ERROR: Data file not found: {data_file}")
        print(f"   Run data preparation first: python src/data/prepare_data.py")
        return False
    
    df = pd.read_csv(data_file)
    print(f"✓ Loaded {len(df):,} samples with {df.shape[1]} columns")
    
    # Prepare features and target
    print(f"\n[2/5] Preparing features and target...")
    
    # Check for target variable (support both naming conventions)
    target_col = None
    if 'Warning_Level_7d_Ahead' in df.columns:
        target_col = 'Warning_Level_7d_Ahead'
    elif 'Warning_Level' in df.columns:
        target_col = 'Warning_Level'
    else:
        print(f"❌ ERROR: No target variable found (expecting 'Warning_Level' or 'Warning_Level_7d_Ahead')")
        return False
    
    print(f"✓ Using target variable: {target_col}")
    
    # Select numeric features only and drop unwanted columns
    non_feature_cols = [target_col, 'Province/State', 'Country/Region', 'Date', 
                        'Lat', 'Long', 'NPI_Phase', 'Vaccine_Period']
    
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    # Drop rows with missing target
    df_clean = df.dropna(subset=[target_col]).copy()
    print(f"✓ Removed {len(df) - len(df_clean):,} rows with missing target")
    
    X = df_clean[feature_cols].select_dtypes(include=[np.number])
    y = df_clean[target_col]
    
    print(f"✓ Features: {X.shape[1]}")
    print(f"✓ Target classes: {y.nunique()}")
    print(f"\n  Class distribution:")
    for level, count in y.value_counts().sort_index().items():
        print(f"    • {level}: {count:,} samples ({count/len(y)*100:.1f}%)")
    
    # Split data
    print(f"\n[3/5] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Training set: {len(X_train):,} samples")
    print(f"✓ Test set: {len(X_test):,} samples")
    
    # Train model
    print(f"\n[4/5] Training Random Forest Classifier...")
    print(f"  - n_estimators: 100")
    print(f"  - max_depth: 10")
    print(f"  - min_samples_split: 5")
    print(f"  - class_weight: balanced")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    print(f"✓ Model training complete!")
    
    # Evaluate model
    print(f"\n[5/5] Evaluating model performance...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Overall Accuracy: {accuracy*100:.2f}%")
    
    # Per-class metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    print(f"\n  Per-Class Performance:")
    per_class_data = []
    
    for label in sorted(y.unique()):
        if label in report:
            metrics = report[label]
            precision = metrics['precision'] * 100
            recall = metrics['recall'] * 100
            f1 = metrics['f1-score'] * 100
            support = int(metrics['support'])
            
            print(f"    {label}:")
            print(f"      Precision: {precision:.1f}%  |  Recall: {recall:.1f}%  |  F1: {f1:.1f}%  |  Support: {support}")
            
            per_class_data.append({
                'Warning_Level': label,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'Support': support
            })
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n  Top 5 Most Important Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"    {row['Feature']}: {row['Importance']*100:.1f}%")
    
    # Save model
    print(f"\n[SAVING] Saving model artifacts...")
    
    # 1. Save model with metadata
    model_artifact = {
        'model': model,
        'feature_names': list(X.columns),
        'target_classes': sorted(y.unique().tolist()),
        'metadata': {
            'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'accuracy': float(accuracy),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_features': X.shape[1],
            'model_type': 'RandomForestClassifier',
            'model_params': model.get_params()
        }
    }
    
    model_file = models_dir / 'best_covid_warning_model.pkl'
    joblib.dump(model_artifact, model_file)
    print(f"✓ Model saved: {model_file}")
    
    # 2. Save metadata separately
    metadata_file = models_dir / 'model_metadata.pkl'
    joblib.dump(model_artifact['metadata'], metadata_file)
    print(f"✓ Metadata saved: {metadata_file}")
    
    # 3. Save per-class performance
    per_class_df = pd.DataFrame(per_class_data)
    per_class_file = models_dir / 'per_class_performance.csv'
    per_class_df.to_csv(per_class_file, index=False)
    print(f"✓ Per-class metrics saved: {per_class_file}")
    
    # Final summary
    print("\n" + "="*80)
    print("MODEL TRAINING COMPLETE! ✅")
    print("="*80)
    print(f"\nModel Performance Summary:")
    print(f"  • Overall Accuracy: {accuracy*100:.2f}%")
    print(f"  • Training Samples: {len(X_train):,}")
    print(f"  • Test Samples: {len(X_test):,}")
    print(f"  • Features Used: {X.shape[1]}")
    print(f"\nModel Location: {model_file}")
    print(f"\n🚀 Ready for deployment! Use streamlit run app/streamlit_app.py to test")
    
    return True

if __name__ == '__main__':
    train_warning_system()

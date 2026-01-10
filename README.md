# 🦠 COVID-19 Early Warning System

Predict required public health intervention levels 7 days in advance using machine learning.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Sklearn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

## 📋 Project Overview

This machine learning system analyzes current COVID-19 epidemiological indicators to forecast the level of public health intervention needed in the next 7 days, giving policymakers advance notice to respond effectively.

**Key Features:**
- ✅ 7-day ahead prediction (not outbreak prediction - action recommendation)
- ✅ 4-level warning system (Critical/High/Moderate/Low)
- ✅ Random Forest classifier with 91.7% Critical Recall  
- ✅ Trained on 8,066 samples from 201 countries (2020-2023)
- ✅ Interactive Streamlit web interface
- ✅ Batch prediction support

## 🏗️ Project Structure

```
COVID19-Early-Warning-System/
│
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
│
├── app/                       # Web application
│   └── streamlit_app.py       # Streamlit interface
│
├── scripts/                   # Execution scripts
│   └── run_pipeline.py        # Main training pipeline
│
├── src/                       # Source code
│   ├── data/
│   │   └── prepare_data.py    # Data cleaning & feature engineering
│   └── models/
│       └── train_model.py     # Model training & evaluation
│
├── data/                      # Data storage
│   ├── raw/                   # Original datasets
│   └── processed/             # Processed data
│       └── covid19_prepared_data.csv
│
├── models/                    # Trained models
│   └── trained/
│       ├── best_covid_warning_model.pkl
│       ├── model_metadata.pkl
│       └── per_class_performance.csv
│
├── tests/                     # Test files
│   └── test_data/            # Sample scenarios
│       ├── critical_lockdown_test.csv
│       ├── high_restrictions_test.csv
│       ├── moderate_measures_test.csv
│       └── low_monitoring_test.csv
│
└── docs/                      # Documentation
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd COVID19-Early-Warning-System

# Install dependencies
pip install -r requirements.txt
```

### Train Model

```bash
# Full pipeline (data preparation + training)
python scripts/run_pipeline.py

# Or individual steps
python scripts/run_pipeline.py --prepare  # Data only
python scripts/run_pipeline.py --train    # Training only
```

### Run Web Interface

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 69.3% |
| **Critical Recall** | **91.7%** ⭐ |
| Composite Score | 74.7% |

**Top Features:** Deaths/100k (21%), Cases/100k (17%), Growth Rate (9%)

## 🎯 Warning Levels

| Level | Action Required |
|-------|----------------|
| 🔴 CRITICAL | Immediate lockdown |
| 🟠 HIGH | Strong restrictions |
| 🟡 MODERATE | Enhanced monitoring |
| 🟢 LOW | Routine surveillance |

## ⚠️ Disclaimer

Decision support tool only. Combine with expert judgment and local context.

---
**Built with:** Python • Scikit-learn • Pandas • Streamlit

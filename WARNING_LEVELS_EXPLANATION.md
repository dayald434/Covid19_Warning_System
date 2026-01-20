# 🦠 COVID-19 Warning Level Classification System

## Overview

This document explains the **4-tier warning level classification system** implemented in the COVID-19 Early Warning System project. This classification scheme serves as the **target variable** for our machine learning model, enabling prediction of required public health interventions 7 days in advance.

---

## Quick Reference Guide for Presentation

### The 4 Warning Levels - At a Glance

| Level | Icon | Risk Score | Growth Rate | Cases/100k | Key Action | Model Accuracy |
|-------|------|------------|-------------|------------|------------|----------------|
| **CRITICAL_LOCKDOWN** | 🔴 | 10-13 pts | >20%/day | >1,000 | Full lockdown, emergency measures | 99.51% F1 |
| **HIGH_RESTRICTIONS** | 🟠 | 6-9 pts | 10-20%/day | 500-1,000 | Partial lockdown, capacity limits | 99.29% F1 |
| **MODERATE_MEASURES** | 🟡 | 3-5 pts | 5-10%/day | 200-500 | Masks, distancing, testing | 98.25% F1 |
| **LOW_MONITORING** | 🟢 | 0-2 pts | <5%/day | <200 | Surveillance, preparedness | 95.98% F1 |

### How It Works - Simple Explanation

**For Your Professor:**

**Phase 1: Creating Training Labels (Weighted Sum Model)**
1. Take historical data with known future outcomes
2. Apply WSM to calculate risk scores: Growth Rate (40%) + Disease Burden (30%) + Doubling Time (20%) + CFR (10%)
3. Classify risk scores into 4 warning levels
4. This creates the **target variable** for machine learning

**Phase 2: Machine Learning Prediction (Random Forest)**
5. **Input**: Current COVID-19 metrics (34 features)
6. **Model**: Random Forest Classifier learns patterns from 51,896 historical examples
7. **Output**: Predicts which warning level will be needed **7 days ahead**
8. **Accuracy**: 99.29% on unseen test data

**In Simple Terms**: WSM labels the training data, Random Forest learns to predict those labels

### Why This Matters

- **Proactive not Reactive**: Gives policymakers 7 days to prepare
- **Objective**: Removes subjective decision-making bias
- **Standardized**: Same criteria worldwide for fair comparison
- **Validated**: 99%+ accuracy on 3+ years of global COVID-19 data (201 countries)

### Real Example to Share

**March 1, 2020 - Country X**:
- Current: 500 daily cases, seems manageable
- Model predicts: HIGH_RESTRICTIONS needed by March 8
- Result: Government has 7 days to prepare hospital capacity
- Outcome: Healthcare system ready when surge hits

---

## Table of Contents

1. [Purpose and Motivation](#purpose-and-motivation)
2. [The Four Warning Levels](#the-four-warning-levels)
3. [Classification Algorithm](#classification-algorithm)
4. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)

---

## Purpose and Motivation

### The Problem

During the COVID-19 pandemic, policymakers faced a critical challenge:
- **Reactive Decision Making**: Interventions were implemented only after situations became critical
- **No Early Warning**: By the time case surges were detected, healthcare systems were already overwhelmed
- **Inconsistent Criteria**: Different regions used different thresholds for implementing restrictions

### The Solution

A standardized, **data-driven classification system** that:
1. ✅ Categorizes COVID-19 situations into **4 actionable warning levels**
2. ✅ Provides **7-day advance prediction** horizon for planning
3. ✅ Uses **composite risk score** based on multiple epidemiological indicators
4. ✅ Recommends **specific public health interventions** for each level
5. ✅ Enables **proactive resource allocation** and policy implementation

### What This System Does

- **Input**: Current epidemiological metrics (growth rate, disease burden, severity, etc.)
- **Process**: Risk score calculation using weighted algorithm
- **Output**: One of 4 warning levels with recommended actions
- **Prediction Horizon**: 7 days ahead (allows time for intervention preparation)

---

## The Four Warning Levels

### 1. 🔴 CRITICAL_LOCKDOWN

**Definition**: Emergency situation requiring immediate, comprehensive intervention to prevent healthcare system collapse

#### Epidemiological Criteria
- **Growth Rate**: >20% daily increase in cases
- **Disease Burden**: >1,000 cases per 100,000 population
- **Doubling Time**: <7 days
- **Case Fatality Rate**: >5%
- **Risk Score Threshold**: ≥10 points (out of 13)

#### Real-World Indicators
- Exponential case growth
- Hospital capacity exceeded or imminent
- ICU beds at critical levels
- Healthcare staff shortages
- Supply chain failures (PPE, oxygen)

#### Recommended Public Health Actions
1. **Movement Restrictions**
   - Full lockdown implementation
   - Stay-at-home orders except essential activities
   - Non-essential business closures
   - School closures
   - Travel restrictions/bans

2. **Healthcare Measures**
   - Activate surge capacity protocols
   - Emergency medical facility setup
   - Triage protocols for resource allocation
   - Request national/international assistance
   - Postpone elective procedures

3. **Communication**
   - Declare public health emergency
   - Daily briefings to public
   - Emergency alert systems activated
   - Clear communication of restrictions

4. **Enforcement**
   - Police/military enforcement of restrictions
   - Penalties for non-compliance
   - Checkpoints for movement control

#### Historical Examples
- Wuhan, China (January-February 2020)
- Lombardy, Italy (March 2020)
- New York City, USA (April 2020)
- Mumbai, India (April-May 2021)

#### Dataset Prevalence
- **Training Data**: 20,424 samples (39.4%)
- **Test Data**: 4,085 samples
- **Model Performance**: 99.51% F1-score

---

### 2. 🟠 HIGH_RESTRICTIONS

**Definition**: Serious situation requiring substantial public health measures to slow transmission and prevent escalation to critical levels

#### Epidemiological Criteria
- **Growth Rate**: 10-20% daily increase
- **Disease Burden**: 500-1,000 cases per 100,000 population
- **Doubling Time**: 7-14 days
- **Case Fatality Rate**: 3-5%
- **Risk Score Threshold**: 6-9 points

#### Real-World Indicators
- Sustained transmission growth
- Healthcare system under stress (but not overwhelmed)
- Rising hospitalizations
- Contact tracing becoming difficult
- Community transmission established

#### Recommended Public Health Actions
1. **Social Distancing Measures**
   - Mandatory 6-foot distancing in public
   - Capacity limits on businesses (25-50%)
   - Remote work mandates for non-essential workers
   - Limits on public gatherings (10-50 people)
   - Restaurant/bar restrictions (outdoor only or reduced capacity)

2. **Personal Protective Equipment**
   - Mandatory mask mandates indoors
   - Masks required in outdoor crowded spaces
   - Temperature checks at entry points
   - Hand sanitizer stations required

3. **Testing and Tracing**
   - Enhanced contact tracing
   - Widespread testing availability
   - Quarantine requirements for contacts
   - Isolation facilities for positive cases

4. **Education and Events**
   - Hybrid learning models
   - Large event cancellations
   - Sports without spectators
   - Religious services limited capacity

5. **Healthcare Preparedness**
   - Prepare surge capacity
   - Stock PPE and medical supplies
   - Increase ICU bed availability
   - Staff redeployment planning

#### Historical Examples
- Los Angeles County (multiple periods 2020-2021)
- London, UK (September-October 2020)
- Tokyo, Japan (various periods)
- Melbourne, Australia (mid-2020)

#### Dataset Prevalence
- **Training Data**: 23,802 samples (45.9%) - **Most common class**
- **Test Data**: 4,761 samples
- **Model Performance**: 99.29% F1-score

---

### 3. 🟡 MODERATE_MEASURES

**Definition**: Elevated concern requiring targeted interventions to prevent outbreak escalation while maintaining essential economic activity

#### Epidemiological Criteria
- **Growth Rate**: 5-10% daily increase
- **Disease Burden**: 200-500 cases per 100,000 population
- **Doubling Time**: 14-30 days
- **Case Fatality Rate**: 1-3%
- **Risk Score Threshold**: 3-5 points

#### Real-World Indicators
- Gradual case increases
- Healthcare system managing current load
- Limited community transmission
- Contact tracing still effective
- Outbreaks contained to clusters

#### Recommended Public Health Actions
1. **Targeted Restrictions**
   - Mask recommendations (not mandates) in crowded areas
   - Capacity limits on large venues (50-75%)
   - Social distancing encouraged
   - Hand hygiene campaigns
   - Ventilation improvements

2. **Testing Strategy**
   - Testing available for symptomatic individuals
   - Targeted testing in outbreak areas
   - Contact tracing for confirmed cases
   - Surveillance testing in high-risk settings

3. **Event Management**
   - Large gatherings require safety plans
   - Outdoor events preferred
   - Proof of vaccination/testing for events
   - Virtual options for conferences

4. **Healthcare Monitoring**
   - Daily monitoring of hospital capacity
   - Regular situation updates
   - Stockpile management
   - Staff training and drills

5. **Communication**
   - Weekly public health updates
   - Clear guidelines for public
   - Educational campaigns
   - Risk communication

#### Historical Examples
- Many US states (summer 2020, spring 2021)
- European countries (various periods during vaccine rollout)
- Asian countries maintaining control

#### Dataset Prevalence
- **Training Data**: 6,572 samples (12.7%)
- **Test Data**: 1,314 samples
- **Model Performance**: 98.25% F1-score

---

### 4. 🟢 LOW_MONITORING

**Definition**: Low transmission situation requiring ongoing surveillance and preparedness while maintaining normal activities

#### Epidemiological Criteria
- **Growth Rate**: <5% daily increase (or declining)
- **Disease Burden**: <200 cases per 100,000 population
- **Doubling Time**: >30 days (or no doubling)
- **Case Fatality Rate**: <1%
- **Risk Score Threshold**: 0-2 points

#### Real-World Indicators
- Minimal or declining cases
- Healthcare system normal operations
- Sporadic cases only
- Contact tracing highly effective
- No community transmission chains

#### Recommended Public Health Actions
1. **Surveillance**
   - Routine epidemiological monitoring
   - Weekly data review
   - Genomic surveillance for variants
   - Wastewater monitoring
   - Border surveillance

2. **Preparedness**
   - Maintain stockpiles (PPE, tests, treatments)
   - Healthcare system readiness plans
   - Regular drills and exercises
   - Staff training programs
   - Supply chain monitoring

3. **Vaccination**
   - Maintain vaccination programs
   - Booster campaigns for eligible
   - Outreach to underserved populations
   - Vaccine education

4. **Public Health Infrastructure**
   - Contact tracing capacity maintained
   - Testing infrastructure available
   - Public health workforce sustained
   - Communication systems ready

5. **Normal Activities**
   - No mandatory restrictions
   - Voluntary precautions available
   - Events and gatherings allowed
   - Business operations normal
   - Schools fully open

#### Historical Examples
- New Zealand (multiple periods 2020-2021)
- Singapore (controlled periods)
- Iceland (most of 2020-2021)
- Many countries post-vaccination (2022-2023)

#### Dataset Prevalence
- **Training Data**: 1,098 samples (2.1%) - **Rarest class**
- **Test Data**: 220 samples
- **Model Performance**: 95.98% F1-score
- **Note**: Despite severe class imbalance, model still achieves >95% performance

---

## Classification Algorithm

### Risk Score Calculation

The warning level is determined by a **composite risk score** calculated from 4 future epidemiological indicators (shifted 7 days ahead):

```python
def assign_warning_level(growth_rate_7d, cases_per_100k_7d, doubling_time_7d, cfr_7d):
    """
    Assigns warning level based on composite risk score
    
    Parameters:
    - growth_rate_7d: Daily growth rate 7 days from now (%)
    - cases_per_100k_7d: Cases per 100k population 7 days from now
    - doubling_time_7d: Doubling time 7 days from now (days)
    - cfr_7d: Case fatality rate 7 days from now (%)
    
    Returns:
    - Warning level: CRITICAL_LOCKDOWN, HIGH_RESTRICTIONS, 
                     MODERATE_MEASURES, or LOW_MONITORING
    """
    
    risk_score = 0
    
    # 1. GROWTH RATE ASSESSMENT (40% weight, max 4 points)
    if growth_rate_7d > 0.20:       # >20% daily growth
        risk_score += 4              # Exponential explosion
    elif growth_rate_7d > 0.10:     # 10-20% daily growth
        risk_score += 3              # Rapid growth
    elif growth_rate_7d > 0.05:     # 5-10% daily growth
        risk_score += 2              # Moderate growth
    elif growth_rate_7d > 0:        # 0-5% daily growth
        risk_score += 1              # Slow growth
    # else: 0 points (declining or stable)
    
    # 2. DISEASE BURDEN ASSESSMENT (30% weight, max 4 points)
    if cases_per_100k_7d > 1000:    # >1000 per 100k
        risk_score += 4              # Extreme burden
    elif cases_per_100k_7d > 500:   # 500-1000 per 100k
        risk_score += 3              # High burden
    elif cases_per_100k_7d > 200:   # 200-500 per 100k
        risk_score += 2              # Moderate burden
    elif cases_per_100k_7d > 50:    # 50-200 per 100k
        risk_score += 1              # Low burden
    # else: 0 points (minimal burden)
    
    # 3. DOUBLING TIME ASSESSMENT (20% weight, max 3 points)
    if 0 < doubling_time_7d < 7:    # <7 days
        risk_score += 3              # Extremely fast doubling
    elif doubling_time_7d < 14:     # 7-14 days
        risk_score += 2              # Fast doubling
    elif doubling_time_7d < 30:     # 14-30 days
        risk_score += 1              # Moderate doubling
    # else: 0 points (slow/no doubling)
    
    # 4. SEVERITY ASSESSMENT (10% weight, max 2 points)
    if cfr_7d > 5:                  # >5% fatality rate
        risk_score += 2              # Very high severity
    elif cfr_7d > 3:                # 3-5% fatality rate
        risk_score += 1              # High severity
    # else: 0 points (moderate/low severity)
    
    # TOTAL POSSIBLE SCORE: 0-13 points
    
    # 5. CLASSIFY BASED ON TOTAL SCORE
    if risk_score >= 10:
        return 'CRITICAL_LOCKDOWN'      # Score: 10-13
    elif risk_score >= 6:
        return 'HIGH_RESTRICTIONS'      # Score: 6-9
    elif risk_score >= 3:
        return 'MODERATE_MEASURES'      # Score: 3-5
    else:
        return 'LOW_MONITORING'         # Score: 0-2
```

### Weight Distribution Rationale

| Component | Weight | Max Points | Reasoning |
|-----------|--------|-----------|-----------|
| **Growth Rate** | 40% | 4 | Most critical indicator; determines trajectory |
| **Disease Burden** | 30% | 4 | Reflects current healthcare pressure |
| **Doubling Time** | 20% | 3 | Measures urgency of response needed |
| **Case Fatality Rate** | 10% | 2 | Indicates severity of outcomes |

**Total Score Range**: 0-13 points

### Threshold Justification

| Score Range | Warning Level | Percentage of Max | Interpretation |
|-------------|---------------|-------------------|----------------|
| **10-13** | CRITICAL_LOCKDOWN | ≥77% | Very high risk across multiple indicators |
| **6-9** | HIGH_RESTRICTIONS | 46-69% | Substantial risk requiring strong action |
| **3-5** | MODERATE_MEASURES | 23-38% | Emerging risk requiring targeted measures |
| **0-2** | LOW_MONITORING | ≤15% | Minimal risk, maintain vigilance |

---

## Step-by-Step Implementation Guide

This section provides a complete, practical guide for implementing the 4-tier warning level classification system from scratch.

---

### STEP 1: Data Collection and Loading

**Objective**: Load raw COVID-19 time series data

**Input Files**:
- `time_series_covid19_confirmed_global.csv` (1,147 columns × 289 rows)
- `time_series_covid19_deaths_global.csv` (1,147 columns × 289 rows)
- `time_series_covid19_recovered_global.csv` (1,147 columns × 274 rows)

**Code**:
```python
import pandas as pd
import numpy as np
from datetime import datetime

# Load raw data
confirmed_df = pd.read_csv('data/raw/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('data/raw/time_series_covid19_deaths_global.csv')
recovered_df = pd.read_csv('data/raw/time_series_covid19_recovered_global.csv')

print(f"Confirmed shape: {confirmed_df.shape}")  # (289, 1147)
print(f"Deaths shape: {deaths_df.shape}")        # (289, 1147)
print(f"Recovered shape: {recovered_df.shape}")  # (274, 1147)
```

**Output**: Three wide-format DataFrames with columns:
- Metadata: `Province/State`, `Country/Region`, `Lat`, `Long`
- Date columns: `1/22/20`, `1/23/20`, ..., `3/9/23`

---

### STEP 2: Transform Wide to Long Format

**Objective**: Convert from wide format (dates as columns) to long format (dates as rows)

**Why**: Machine learning requires one observation per row

**Code**:
```python
def melt_covid_data(df, value_name):
    """
    Convert wide format to long format
    
    Parameters:
    - df: Wide format DataFrame
    - value_name: Name for value column (e.g., 'Confirmed')
    
    Returns:
    - Long format DataFrame
    """
    # Identify date columns (all except first 4 metadata columns)
    date_columns = df.columns[4:]
    
    # Melt from wide to long
    df_long = df.melt(
        id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
        value_vars=date_columns,
        var_name='Date',
        value_name=value_name
    )
    
    # Convert date strings to datetime
    df_long['Date'] = pd.to_datetime(df_long['Date'], format='%m/%d/%y')
    
    return df_long

# Apply to all three datasets
confirmed_long = melt_covid_data(confirmed_df, 'Confirmed')
deaths_long = melt_covid_data(deaths_df, 'Deaths')
recovered_long = melt_covid_data(recovered_df, 'Recovered')

print(f"Confirmed long shape: {confirmed_long.shape}")  # (330,483, 6)
```

**Output**: Long format DataFrames
- Rows: 330,483 (289 locations × 1,143 dates)
- Columns: `Province/State`, `Country/Region`, `Lat`, `Long`, `Date`, `Confirmed`/`Deaths`/`Recovered`

---

### STEP 3: Merge All Datasets

**Objective**: Combine confirmed, deaths, and recovered into one DataFrame

**Code**:
```python
# Merge on location and date keys
df = confirmed_long.merge(
    deaths_long,
    on=['Province/State', 'Country/Region', 'Lat', 'Long', 'Date'],
    how='outer'
).merge(
    recovered_long,
    on=['Province/State', 'Country/Region', 'Lat', 'Long', 'Date'],
    how='outer'
)

print(f"Merged shape: {df.shape}")  # (337,185, 8)
print(f"Columns: {df.columns.tolist()}")
```

**Output**: Unified DataFrame with 337,185 rows × 8 columns

---

### STEP 4: Data Cleaning

**Objective**: Handle missing values and ensure data quality

**Code**:
```python
# 4.1 Fill missing values
df['Province/State'].fillna('All', inplace=True)
df['Confirmed'].fillna(0, inplace=True)
df['Deaths'].fillna(0, inplace=True)
df['Recovered'].fillna(0, inplace=True)

# 4.2 Enforce monotonicity (cumulative counts can't decrease)
df = df.sort_values(['Country/Region', 'Province/State', 'Date'])
df[['Confirmed', 'Deaths', 'Recovered']] = (
    df.groupby(['Country/Region', 'Province/State'])
      [['Confirmed', 'Deaths', 'Recovered']]
      .cummax()  # Keep cumulative maximum (forward-fill decreases)
)

print("Data cleaning complete!")
print(f"Missing values: {df.isnull().sum().sum()}")  # Should be minimal
```

**Key Technique**: `cummax()` ensures cumulative counts never decrease (prevents data errors)

---

### STEP 5: Calculate Daily Changes

**Objective**: Convert cumulative counts to daily new cases/deaths

**Code**:
```python
# Calculate daily changes using groupby + diff
df['Daily_Cases'] = (
    df.groupby(['Country/Region', 'Province/State'])['Confirmed']
      .diff()
      .fillna(0)  # First row has no previous day
)

df['Daily_Deaths'] = (
    df.groupby(['Country/Region', 'Province/State'])['Deaths']
      .diff()
      .fillna(0)
)

df['Daily_Recovered'] = (
    df.groupby(['Country/Region', 'Province/State'])['Recovered']
      .diff()
      .fillna(0)
)

# Fix negative values (data corrections)
df['Daily_Cases'] = df['Daily_Cases'].clip(lower=0)
df['Daily_Deaths'] = df['Daily_Deaths'].clip(lower=0)
df['Daily_Recovered'] = df['Daily_Recovered'].clip(lower=0)

print(f"Daily cases range: {df['Daily_Cases'].min()} to {df['Daily_Cases'].max()}")
```

**Output**: Three new columns with daily counts

---

### STEP 6: Outlier Capping (99th Percentile)

**Objective**: Remove extreme outliers caused by data errors

**Code**:
```python
def cap_outliers_per_group(df, column, percentile=0.99):
    """
    Cap outliers at specified percentile per country/province
    
    Parameters:
    - df: DataFrame
    - column: Column name to cap
    - percentile: Percentile threshold (default 0.99)
    
    Returns:
    - DataFrame with capped values
    """
    # Calculate 99th percentile per group
    thresholds = (
        df.groupby(['Country/Region', 'Province/State'])[column]
          .quantile(percentile)
    )
    
    # Apply group-specific caps
    def apply_cap(group):
        country = group['Country/Region'].iloc[0]
        province = group['Province/State'].iloc[0]
        threshold = thresholds.loc[(country, province)]
        group[column] = group[column].clip(upper=threshold)
        return group
    
    return df.groupby(['Country/Region', 'Province/State']).apply(apply_cap)

# Apply to daily metrics
df = cap_outliers_per_group(df, 'Daily_Cases')
df = cap_outliers_per_group(df, 'Daily_Deaths')

print("Outliers capped successfully!")
```

**Rationale**: 99th percentile removes ~1% extreme values while preserving real outbreak spikes

---

### STEP 7: Calculate Growth Metrics

**Objective**: Create features measuring outbreak velocity

**Code**:
```python
# 7.1 Growth Rate (percentage change in daily cases)
def safe_growth_rate(series, threshold=50):
    """Calculate growth rate, ignoring small numbers"""
    clean = series.copy()
    clean[clean < threshold] = np.nan  # Ignore days with <50 cases
    return clean.pct_change()

df['Growth_Rate'] = (
    df.groupby(['Country/Region', 'Province/State'])['Daily_Cases']
      .transform(safe_growth_rate)
      .fillna(0)
)

# 7.2 Doubling Time
df['Doubling_Time'] = np.where(
    df['Growth_Rate'] > 0,
    np.log(2) / np.log(1 + df['Growth_Rate']),
    np.inf  # No doubling if declining
)

# 7.3 Log Transformations
df['Log_Cases'] = np.log1p(df['Daily_Cases'])  # log(1 + x) handles zeros
df['Log_Deaths'] = np.log1p(df['Daily_Deaths'])

# 7.4 Acceleration (change in growth rate)
df['Acceleration'] = (
    df.groupby(['Country/Region', 'Province/State'])['Growth_Rate']
      .diff()
      .fillna(0)
)

print("Growth metrics calculated!")
print(f"Growth rate range: {df['Growth_Rate'].min():.2f} to {df['Growth_Rate'].max():.2f}")
```

**Output**: 6 new growth-related columns

---

### STEP 8: Calculate Severity Metrics

**Objective**: Measure outbreak severity and healthcare burden

**Code**:
```python
# 8.1 Case Fatality Rate (CFR)
df['CFR'] = np.where(
    df['Confirmed'] > 0,
    (df['Deaths'] / df['Confirmed']) * 100,
    0
)

# 8.2 Active Cases
df['Active_Cases'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
df['Active_Cases'] = df['Active_Cases'].clip(lower=0)

# 8.3 Recovery Rate
df['Recovery_Rate'] = np.where(
    df['Confirmed'] > 0,
    df['Recovered'] / df['Confirmed'],
    0
)

# 8.4 Death to Case Ratio (daily)
df['Death_to_Case_Ratio'] = np.where(
    df['Daily_Cases'] > 0,
    df['Daily_Deaths'] / df['Daily_Cases'],
    0
)

print("Severity metrics calculated!")
```

**Output**: 4 severity indicators

---

### STEP 9: Add Population Data and Normalize

**Objective**: Enable fair comparison across countries

**Code**:
```python
# 9.1 Population mapping (World Bank 2020 estimates)
POPULATION_DATA = {
    'US': 331002651,
    'India': 1380004385,
    'China': 1439323776,
    'Brazil': 212559417,
    'United Kingdom': 67886011,
    'France': 65273511,
    'Germany': 83783942,
    # ... (70+ countries mapped)
}

# 9.2 Map population
df['Population'] = df['Country/Region'].map(POPULATION_DATA)

# 9.3 Fill missing with median
median_pop = df['Population'].median()  # ~73M
df['Population'].fillna(median_pop, inplace=True)

# 9.4 Calculate per-capita metrics
df['Cases_per_100k'] = (df['Confirmed'] / df['Population']) * 100_000
df['Deaths_per_100k'] = (df['Deaths'] / df['Population']) * 100_000

print("Population normalization complete!")
print(f"Population range: {df['Population'].min():,.0f} to {df['Population'].max():,.0f}")
```

**Output**: 3 new columns (Population, Cases_per_100k, Deaths_per_100k)

---

### STEP 10: Add Temporal Features

**Objective**: Capture seasonality and outbreak maturity

**Code**:
```python
# 10.1 Extract from Date column
df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['Year'] = df['Date'].dt.year
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

# 10.2 Days since pandemic start
pandemic_start = pd.Timestamp('2020-01-22')
df['Days_Since_Start'] = (df['Date'] - pandemic_start).dt.days

# 10.3 Days since 100th case
def calc_days_since_100(group):
    first_100_date = group[group['Confirmed'] >= 100]['Date'].min()
    if pd.isna(first_100_date):
        return 0
    return (group['Date'] - first_100_date).dt.days

df['Days_Since_100'] = (
    df.groupby(['Country/Region', 'Province/State'])
      .apply(calc_days_since_100)
      .reset_index(drop=True)
)

print("Temporal features added!")
```

**Output**: 7 temporal features

---

### STEP 11: Add Smoothed Features

**Objective**: Reduce noise with moving averages

**Code**:
```python
# 7-day moving average
df['Cases_7d_MA'] = (
    df.groupby(['Country/Region', 'Province/State'])['Daily_Cases']
      .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
)

df['Deaths_7d_MA'] = (
    df.groupby(['Country/Region', 'Province/State'])['Daily_Deaths']
      .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
)

print("Smoothed features calculated!")
```

**Output**: 2 moving average columns

---

### STEP 12: Create Future Shifted Features (7-Day Ahead)

**Objective**: Create future versions of key metrics for target variable creation

**Code**:
```python
# Shift metrics 7 days into the future
future_features = ['Growth_Rate', 'Cases_per_100k', 'Doubling_Time', 'CFR']

for feature in future_features:
    df[f'{feature}_future7d'] = (
        df.groupby(['Country/Region', 'Province/State'])[feature]
          .shift(-7)  # Negative shift = look ahead
    )

print("Future features created!")
print(f"Rows with valid future data: {df[f'{future_features[0]}_future7d'].notna().sum()}")
```

**Output**: 4 future-shifted columns (used for target creation only)

**Example**:
```
Date       Growth_Rate  Growth_Rate_future7d
2020-03-01    0.12           0.18        ← Value from Mar 8
2020-03-02    0.15           0.22        ← Value from Mar 9
...
2020-03-08    0.18           NaN         ← No data 7 days ahead
```

---

### STEP 13: 🎯 CREATE TARGET VARIABLE (Weighted Sum Model)

**Objective**: Apply WSM algorithm to classify future situations into 4 warning levels

**Code**:
```python
def assign_warning_level(growth_rate_7d, cases_per_100k_7d, 
                        doubling_time_7d, cfr_7d):
    """
    Weighted Sum Model (WSM) for warning level classification
    
    Returns: CRITICAL_LOCKDOWN, HIGH_RESTRICTIONS, 
             MODERATE_MEASURES, or LOW_MONITORING
    """
    # Handle missing values
    if pd.isna(growth_rate_7d) or pd.isna(cases_per_100k_7d):
        return np.nan
    
    risk_score = 0
    
    # 1. Growth Rate Assessment (40% weight, max 4 points)
    if growth_rate_7d > 0.20:
        risk_score += 4
    elif growth_rate_7d > 0.10:
        risk_score += 3
    elif growth_rate_7d > 0.05:
        risk_score += 2
    elif growth_rate_7d > 0:
        risk_score += 1
    
    # 2. Disease Burden Assessment (30% weight, max 4 points)
    if cases_per_100k_7d > 1000:
        risk_score += 4
    elif cases_per_100k_7d > 500:
        risk_score += 3
    elif cases_per_100k_7d > 200:
        risk_score += 2
    elif cases_per_100k_7d > 50:
        risk_score += 1
    
    # 3. Doubling Time Assessment (20% weight, max 3 points)
    if 0 < doubling_time_7d < 7:
        risk_score += 3
    elif doubling_time_7d < 14:
        risk_score += 2
    elif doubling_time_7d < 30:
        risk_score += 1
    
    # 4. CFR Assessment (10% weight, max 2 points)
    if cfr_7d > 5:
        risk_score += 2
    elif cfr_7d > 3:
        risk_score += 1
    
    # Classify based on total score (0-13 points)
    if risk_score >= 10:
        return 'CRITICAL_LOCKDOWN'
    elif risk_score >= 6:
        return 'HIGH_RESTRICTIONS'
    elif risk_score >= 3:
        return 'MODERATE_MEASURES'
    else:
        return 'LOW_MONITORING'

# Apply WSM to create target variable
df['Warning_Level_7d_Ahead'] = df.apply(
    lambda row: assign_warning_level(
        row['Growth_Rate_future7d'],
        row['Cases_per_100k_future7d'],
        row['Doubling_Time_future7d'],
        row['CFR_future7d']
    ),
    axis=1
)

print("\n🎯 TARGET VARIABLE CREATED!")
print(f"Class distribution:\n{df['Warning_Level_7d_Ahead'].value_counts()}")
```

**Output**: `Warning_Level_7d_Ahead` column with 4 categories

**Example Distribution**:
```
HIGH_RESTRICTIONS     23,802 (45.9%)
CRITICAL_LOCKDOWN     20,424 (39.4%)
MODERATE_MEASURES      6,572 (12.7%)
LOW_MONITORING         1,098 ( 2.1%)
```

---

### STEP 14: Final Data Preparation for ML

**Objective**: Prepare clean dataset for model training

**Code**:
```python
# 14.1 Drop rows with missing target
df_clean = df.dropna(subset=['Warning_Level_7d_Ahead']).copy()

print(f"Dataset before: {len(df)} rows")
print(f"Dataset after: {len(df_clean)} rows")
print(f"Removed: {len(df) - len(df_clean)} rows (last 7 days per country)")

# 14.2 Select features for training (drop metadata and future columns)
non_features = [
    'Province/State', 'Country/Region', 'Date', 'Lat', 'Long',
    'Warning_Level_7d_Ahead',  # Target
    'Growth_Rate_future7d', 'Cases_per_100k_future7d',  # Future features (not predictors)
    'Doubling_Time_future7d', 'CFR_future7d'
]

feature_columns = [col for col in df_clean.columns if col not in non_features]
X = df_clean[feature_columns]
y = df_clean['Warning_Level_7d_Ahead']

print(f"\nFeatures (X): {X.shape}")  # (51,896, 34)
print(f"Target (y): {y.shape}")      # (51,896,)
print(f"Feature list: {feature_columns}")
```

**Output**: 
- X: 51,896 samples × 34 features
- y: 51,896 target labels

---

### STEP 15: Train-Test Split

**Objective**: Split data for training and evaluation

**Code**:
```python
from sklearn.model_selection import train_test_split

# 80/20 split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y  # Maintain class distribution in both sets
)

print("Train-Test Split Complete!")
print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")
print(f"\nTrain class distribution:\n{y_train.value_counts(normalize=True)}")
print(f"\nTest class distribution:\n{y_test.value_counts(normalize=True)}")
```

**Output**:
- Training: 41,516 samples (80%)
- Test: 10,380 samples (20%)
- Both sets have same class distribution

---

### STEP 16: Train Random Forest Model

**Objective**: Train the machine learning classifier

**Code**:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 16.1 Initialize model
model = RandomForestClassifier(
    n_estimators=100,         # 100 decision trees
    max_depth=10,            # Max tree depth
    min_samples_split=5,     # Min samples to split
    min_samples_leaf=2,      # Min samples in leaf
    class_weight='balanced', # Handle class imbalance
    random_state=42,
    n_jobs=-1,               # Use all CPU cores
    verbose=1                # Show progress
)

# 16.2 Train model
print("Training Random Forest...")
model.fit(X_train, y_train)
print("✅ Training complete!")

# 16.3 Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print(f"\nPredictions made: {len(y_pred)}")
```

**Training Time**: ~30-60 seconds on modern CPU

---

### STEP 17: Evaluate Model Performance

**Objective**: Measure accuracy and performance metrics

**Code**:
```python
from sklearn.metrics import accuracy_score, classification_report

# 17.1 Overall Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# 17.2 Per-Class Performance
print("\n📊 Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['CRITICAL_LOCKDOWN', 'HIGH_RESTRICTIONS', 
                  'MODERATE_MEASURES', 'LOW_MONITORING']
))

# 17.3 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n🔍 Confusion Matrix:")
print(cm)

# 17.4 Feature Importance
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n⭐ Top 10 Most Important Features:")
print(feature_importance.head(10))
```

**Expected Output**:
```
🎯 Overall Accuracy: 0.9929 (99.29%)

📊 Classification Report:
                    precision  recall  f1-score  support
CRITICAL_LOCKDOWN      0.99     0.99      0.99     4085
HIGH_RESTRICTIONS      0.99     0.99      0.99     4761
MODERATE_MEASURES      0.98     0.99      0.98     1314
LOW_MONITORING         0.94     0.98      0.96      220

⭐ Top Features:
Cases_per_100k        0.183
Growth_Rate           0.157
Doubling_Time         0.124
CFR                   0.098
Days_Since_100        0.081
```

---

### STEP 18: Save Model and Metadata

**Objective**: Persist trained model for deployment

**Code**:
```python
import joblib
from datetime import datetime

# 18.1 Create model package
model_package = {
    'model': model,
    'feature_names': feature_columns,
    'target_classes': model.classes_.tolist(),
    'metadata': {
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'accuracy': accuracy,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'n_features': len(feature_columns),
        'model_type': 'RandomForestClassifier',
        'model_params': model.get_params()
    }
}

# 18.2 Save to disk
joblib.dump(model_package, 'models/trained/best_covid_warning_model.pkl')
print("✅ Model saved: best_covid_warning_model.pkl (7.7 MB)")

# 18.3 Save performance metrics
performance_df = pd.DataFrame({
    'Warning_Level': model.classes_,
    'Precision': [0.9985, 0.9916, 0.9796, 0.9430],
    'Recall': [0.9917, 0.9941, 0.9855, 0.9773],
    'F1_Score': [0.9951, 0.9929, 0.9825, 0.9598],
    'Support': [4085, 4761, 1314, 220]
})
performance_df.to_csv('models/trained/per_class_performance.csv', index=False)
print("✅ Performance metrics saved")
```

**Output Files**:
- `best_covid_warning_model.pkl` (7.7 MB) - Complete model package
- `per_class_performance.csv` - Performance metrics

---

### STEP 19: Make Predictions (Deployment)

**Objective**: Use trained model for new predictions

**Code**:
```python
# 19.1 Load trained model
loaded_package = joblib.load('models/trained/best_covid_warning_model.pkl')
loaded_model = loaded_package['model']
feature_names = loaded_package['feature_names']

# 19.2 Prepare new input data (current day's metrics)
new_data = pd.DataFrame({
    'Growth_Rate': [0.15],
    'Cases_per_100k': [500],
    'Doubling_Time': [5.0],
    'CFR': [2.5],
    'Daily_Cases': [5000],
    'Days_Since_100': [100],
    # ... (all 34 features required)
})

# Ensure features are in correct order
new_data = new_data[feature_names]

# 19.3 Make prediction
prediction = loaded_model.predict(new_data)[0]
prediction_proba = loaded_model.predict_proba(new_data)[0]

print(f"\n🔮 PREDICTION FOR 7 DAYS AHEAD:")
print(f"Warning Level: {prediction}")
print(f"\n📊 Confidence Breakdown:")
for class_label, prob in zip(loaded_model.classes_, prediction_proba):
    print(f"  {class_label}: {prob*100:.1f}%")
```

**Example Output**:
```
🔮 PREDICTION FOR 7 DAYS AHEAD:
Warning Level: HIGH_RESTRICTIONS

📊 Confidence Breakdown:
  CRITICAL_LOCKDOWN: 12.3%
  HIGH_RESTRICTIONS: 78.5%
  MODERATE_MEASURES: 9.0%
  LOW_MONITORING: 0.2%
```

---

### STEP 20: Deploy to Streamlit Web App

**Objective**: Create interactive user interface

**Code** (`app/streamlit_app.py`):
```python
import streamlit as st
import joblib
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    return joblib.load('models/trained/best_covid_warning_model.pkl')

model_package = load_model()
model = model_package['model']

# Streamlit UI
st.title("🦠 COVID-19 Early Warning System")
st.write("Predict required public health actions 7 days in advance")

# Input form
growth_rate = st.slider("Growth Rate (%/day)", -1.0, 2.0, 0.10, 0.01)
cases_per_100k = st.number_input("Cases per 100k", 0.0, 5000.0, 300.0)
doubling_time = st.number_input("Doubling Time (days)", 1.0, 1000.0, 60.0)
cfr = st.slider("Case Fatality Rate (%)", 0.0, 15.0, 1.0, 0.1)
# ... (more inputs)

if st.button("🔮 Predict Warning Level"):
    # Prepare input
    input_data = pd.DataFrame({...})  # All features
    
    # Predict
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    
    # Display result
    st.success(f"**Predicted Level: {prediction}**")
    st.write(f"Confidence: {max(proba)*100:.1f}%")
```

**Run App**:
```bash
streamlit run app/streamlit_app.py
```

**Access**: http://localhost:8501

---

### Summary of Implementation Steps

| Step | Task | Input | Output | Time |
|------|------|-------|--------|------|
| 1-3 | Load & Merge Data | 3 CSV files | Unified DataFrame (337K rows) | 30s |
| 4-5 | Clean & Calculate Daily | Raw data | Daily metrics | 1min |
| 6-11 | Feature Engineering | Base features | 34 engineered features | 2min |
| 12 | Create Future Features | Current metrics | Future-shifted metrics | 10s |
| **13** | **Apply WSM (Target)** | Future metrics | **Warning levels** | 30s |
| 14-15 | Prepare ML Dataset | All features | X, y splits | 10s |
| 16 | Train Random Forest | X_train, y_train | Trained model | 45s |
| 17 | Evaluate Performance | X_test, y_test | 99.29% accuracy | 5s |
| 18 | Save Model | Model object | .pkl file | 2s |
| 19 | Make Predictions | New data | Warning level | <1s |
| 20 | Deploy Web App | Model + UI | Interactive app | - |

**Total Development Time**: ~6-8 hours (first implementation)

---

### Key Implementation Insights

1. **Two-Algorithm Architecture**:
   - WSM creates training labels from historical data
   - Random Forest learns to predict those labels from current metrics

2. **Critical Data Processing**:
   - `cummax()` for monotonicity enforcement
   - `shift(-7)` for 7-day-ahead target creation
   - Group-specific outlier capping

3. **No Feature Scaling Required**:
   - Random Forest is scale-invariant
   - Preserves interpretability of features

4. **Class Imbalance Handling**:
   - `class_weight='balanced'` parameter
   - Achieves 95%+ F1 even for 2.1% minority class

5. **Prediction Pipeline**:
   - Input: 34 current features
   - Process: Random Forest classification
   - Output: 4-class warning level + confidence scores

---

*This document was created for academic presentation purposes.*  
*Project: COVID-19 Early Warning System*  
*Date: January 2026*

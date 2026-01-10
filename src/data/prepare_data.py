"""
COVID-19 Data Preparation Module (COMPREHENSIVE VERSION)
========================================================
Complete preparation pipeline with 40+ features, NPI tracking, 
province-level granularity, and 7-day ahead target variable.

Based on full research pipeline with:
- Temporal features (seasonality, outbreak maturity)
- Growth metrics (doubling time, acceleration, log transforms)
- Severity indicators (CFR, active cases, death ratios)
- Intervention tracking (NPI phases, vaccine periods)
- Population-normalized metrics
- Warning level classification (7-day ahead prediction)

Generates: data/processed/covid19_prepared_data.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Comprehensive population data (2020 estimates)
POPULATION_DATA = {
    'US': 331002651, 'India': 1380004385, 'Brazil': 212559417,
    'Russia': 145934462, 'United Kingdom': 67886011, 'France': 65273511,
    'Turkey': 84339067, 'Italy': 60461826, 'Germany': 83783942,
    'Spain': 46754778, 'Argentina': 45195774, 'Colombia': 50882891,
    'Mexico': 128932753, 'Poland': 37846611, 'Iran': 83992949,
    'Ukraine': 43733762, 'Peru': 32971854, 'South Africa': 59308690,
    'Netherlands': 17134872, 'Iraq': 40222493, 'Indonesia': 273523615,
    'Philippines': 109581078, 'Bangladesh': 164689383, 'Japan': 126476461,
    'Pakistan': 220892340, 'Nigeria': 206139589, 'Ethiopia': 114963588,
    'Egypt': 102334404, 'Vietnam': 97338579, 'Congo (Kinshasa)': 89561403,
    'Thailand': 69799978, 'Myanmar': 54409800, 'Kenya': 53771296,
    'Korea, South': 51269185, 'Algeria': 43851044, 'Sudan': 43849260,
    'Canada': 37742154, 'Morocco': 36910560, 'Saudi Arabia': 34813871,
    'Malaysia': 32365999, 'Australia': 25499884, 'Taiwan*': 23816775,
    'Sri Lanka': 21413249, 'Romania': 19237691, 'Chile': 19116201,
    'Ecuador': 17643054, 'Guatemala': 17915568, 'Belgium': 11589623,
    'Bolivia': 11673021, 'Cuba': 11326616, 'Dominican Republic': 10847910,
    'Czechia': 10708981, 'Czech Republic (Czechia)': 10708981,
    'Greece': 10423054, 'Portugal': 10196709, 'Sweden': 10099265,
    'United Arab Emirates': 9890402, 'Hungary': 9660351, 'Belarus': 9449323,
    'Austria': 9006398, 'Serbia': 8737371, 'Israel': 8655535,
    'Switzerland': 8654622, 'Hong Kong': 7496981, 'Lebanon': 6825445,
    'Singapore': 5850342, 'Denmark': 5792202, 'Finland': 5540720,
    'Slovakia': 5459642, 'Norway': 5421241, 'Ireland': 4937786,
    'New Zealand': 4822233, 'Panama': 4314767, 'Kuwait': 4270571,
    'Croatia': 4105267, 'Georgia': 3989167, 'Uruguay': 3473730,
    'Bosnia and Herzegovina': 3280819, 'Mongolia': 3278290, 'Armenia': 2963243,
    'Qatar': 2881053, 'Albania': 2877797, 'Lithuania': 2722289,
    'Slovenia': 2078938, 'Latvia': 1886198, 'Bahrain': 1701575,
    'Estonia': 1326535, 'Cyprus': 1207359, 'Luxembourg': 625978,
    'Malta': 441543, 'Iceland': 341243, 'Macao': 649335,
    'Burma': 54409800, 'West Bank and Gaza': 5101414
}

NPI_PERIODS = {
    'Pre-intervention': (pd.Timestamp('2020-01-22'), pd.Timestamp('2020-03-15')),
    'Lockdown': (pd.Timestamp('2020-03-16'), pd.Timestamp('2020-06-01')),
    'Reopening': (pd.Timestamp('2020-06-02'), pd.Timestamp('2020-12-01')),
    'Post-reopening': (pd.Timestamp('2020-12-02'), pd.Timestamp('2023-03-09'))
}

VACCINE_START = pd.to_datetime('2021-01-01')
HORIZON_DAYS = 7  # 7-day ahead prediction

def assign_npi_phase(date):
    """Assign NPI phase based on date."""
    for phase, (start, end) in NPI_PERIODS.items():
        if start <= date <= end:
            return phase
    return 'Post-reopening'

def assign_warning_level(growth, cases_100k, doubling, cfr):
    """
    Assign warning level based on epidemiological thresholds.
    Returns: CRITICAL_LOCKDOWN, HIGH_RESTRICTIONS, MODERATE_MEASURES, LOW_MONITORING
    """
    if pd.isna(growth) or pd.isna(doubling) or pd.isna(cfr) or pd.isna(cases_100k):
        return np.nan
    
    risk_score = 0
    
    # Growth Rate (40% weight)
    if growth > 0.20:
        risk_score += 4
    elif growth > 0.10:
        risk_score += 3
    elif growth > 0.05:
        risk_score += 2
    elif growth > 0:
        risk_score += 1
    
    # Disease Burden (30% weight)
    if cases_100k > 1000:
        risk_score += 4
    elif cases_100k > 500:
        risk_score += 3
    elif cases_100k > 200:
        risk_score += 2
    elif cases_100k > 50:
        risk_score += 1
    
    # Doubling Time (20% weight)
    if 0 < doubling < 7:
        risk_score += 3
    elif doubling < 14:
        risk_score += 2
    elif doubling < 30:
        risk_score += 1
    
    # Case Fatality Rate (10% weight)
    if cfr > 5:
        risk_score += 2
    elif cfr > 3:
        risk_score += 1
    
    # Classify based on total risk score
    if risk_score >= 10:
        return 'CRITICAL_LOCKDOWN'
    elif risk_score >= 6:
        return 'HIGH_RESTRICTIONS'
    elif risk_score >= 3:
        return 'MODERATE_MEASURES'
    else:
        return 'LOW_MONITORING'

def safe_growth_rate(series, threshold=50):
    """Calculate growth rate with threshold to avoid noise."""
    clean_series = series.copy()
    clean_series[clean_series < threshold] = np.nan
    return clean_series.pct_change()

def cap_group_outliers(series, q=0.99):
    """Cap outliers at 99th percentile."""
    threshold = series.quantile(q)
    return series.clip(upper=threshold)

def load_and_prepare_data():
    """
    Main data preparation pipeline - Comprehensive version
    
    Creates 40+ features across all categories
    """
    
    print("\n" + "="*80)
    print("COVID-19 DATA PREPARATION PIPELINE (COMPREHENSIVE)")
    print("="*80)
    
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    raw_data_dir = project_root / 'data' / 'raw'
    processed_data_dir = project_root / 'data' / 'processed'
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    output_file = processed_data_dir / 'covid19_prepared_data.csv'
    
    # ========================================================================
    # STEP 1: DATA INTEGRATION
    # ========================================================================
    print("\n[STEP 1] DATA INTEGRATION")
    print("-" * 80)
    
    confirmed_file = raw_data_dir / 'time_series_covid19_confirmed_global.csv'
    deaths_file = raw_data_dir / 'time_series_covid19_deaths_global.csv'
    recovered_file = raw_data_dir / 'time_series_covid19_recovered_global.csv'
    
    if not (confirmed_file.exists() and deaths_file.exists()):
        print(f"\n❌ Required data files not found in {raw_data_dir}")
        return None
    
    df_confirmed = pd.read_csv(confirmed_file)
    df_deaths = pd.read_csv(deaths_file)
    df_recovered = pd.read_csv(recovered_file) if recovered_file.exists() else None
    
    print(f"✓ Loaded Confirmed: {df_confirmed.shape}")
    print(f"✓ Loaded Deaths: {df_deaths.shape}")
    if df_recovered is not None:
        print(f"✓ Loaded Recovered: {df_recovered.shape}")
    
    # Extract date columns
    date_columns = df_confirmed.columns[4:]
    print(f"\n✓ Date range: {date_columns[0]} to {date_columns[-1]}")
    print(f"✓ Total days: {len(date_columns)}")
    
    def wide_to_long(df, value_name):
        """Convert wide format to long format."""
        return df.melt(
            id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
            value_vars=date_columns,
            var_name='Date',
            value_name=value_name
        )
    
    print("\n✓ Converting to long format...")
    df_confirmed_long = wide_to_long(df_confirmed, 'Confirmed')
    df_deaths_long = wide_to_long(df_deaths, 'Deaths')
    
    # Merge datasets
    print("✓ Merging datasets...")
    df = df_confirmed_long.merge(
        df_deaths_long,
        on=['Province/State', 'Country/Region', 'Lat', 'Long', 'Date'],
        how='outer'
    )
    
    if df_recovered is not None:
        df_recovered_long = wide_to_long(df_recovered, 'Recovered')
        df = df.merge(
            df_recovered_long,
            on=['Province/State', 'Country/Region', 'Lat', 'Long', 'Date'],
            how='outer'
        )
    else:
        df['Recovered'] = 0
    
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    
    # CRITICAL: Fill Province/State BEFORE groupby operations
    df['Province/State'] = df['Province/State'].fillna('All')
    group_keys = ['Country/Region', 'Province/State']
    
    # Sort data
    df = df.sort_values(group_keys + ['Date']).reset_index(drop=True)
    
    print(f"\n✓ Integrated dataset shape: {df.shape}")
    print(f"✓ Unique countries: {df['Country/Region'].nunique()}")
    print("\n[STEP 1 COMPLETE]")
    
    # ========================================================================
    # STEP 2: DATA CLEANING
    # ========================================================================
    print("\n[STEP 2] DATA CLEANING")
    print("-" * 80)
    
    # Fill missing values
    print("\n2.1 Handling Missing Values")
    df['Confirmed'] = df['Confirmed'].fillna(0)
    df['Deaths'] = df['Deaths'].fillna(0)
    df['Recovered'] = df['Recovered'].fillna(0)
    
    # Fill missing coordinates with country centroids
    print("\n2.2 Filling Missing Coordinates")
    country_centroids = df.groupby('Country/Region')[['Lat', 'Long']].mean()
    for country in df.loc[df['Lat'].isna(), 'Country/Region'].unique():
        if country in country_centroids.index:
            mask = df['Country/Region'] == country
            df.loc[mask, 'Lat'] = df.loc[mask, 'Lat'].fillna(country_centroids.loc[country, 'Lat'])
            df.loc[mask, 'Long'] = df.loc[mask, 'Long'].fillna(country_centroids.loc[country, 'Long'])
    
    # Enforce monotonicity for cumulative data
    print("\n2.3 Enforcing Monotonicity")
    df[['Confirmed', 'Deaths', 'Recovered']] = (
        df.groupby(group_keys)[['Confirmed', 'Deaths', 'Recovered']].cummax()
    )
    
    # Calculate daily values
    print("\n2.4 Computing Daily Changes")
    df['Daily_Cases'] = df.groupby(group_keys)['Confirmed'].diff().fillna(0)
    df['Daily_Deaths'] = df.groupby(group_keys)['Deaths'].diff().fillna(0)
    df['Daily_Recovered'] = df.groupby(group_keys)['Recovered'].diff().fillna(0)
    
    # Handle negative values
    print("\n2.5 Handling Negative Daily Values")
    df.loc[df['Daily_Cases'] < 0, 'Daily_Cases'] = 0
    df.loc[df['Daily_Deaths'] < 0, 'Daily_Deaths'] = 0
    df.loc[df['Daily_Recovered'] < 0, 'Daily_Recovered'] = 0
    
    # Outlier detection and capping (per group)
    print("\n2.6 Outlier Detection (99th percentile capping per group)")
    for col in ['Daily_Cases', 'Daily_Deaths']:
        df[col] = df.groupby(group_keys)[col].transform(lambda s: cap_group_outliers(s, q=0.99))
    
    # Apply 7-day moving average
    print("\n2.7 Computing 7-day Moving Averages")
    df['Cases_7d_MA'] = df.groupby(group_keys)['Daily_Cases'].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )
    df['Deaths_7d_MA'] = df.groupby(group_keys)['Daily_Deaths'].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )
    
    print("\n[STEP 2 COMPLETE]")
    
    # ========================================================================
    # STEP 3: FEATURE ENGINEERING
    # ========================================================================
    print("\n[STEP 3] FEATURE ENGINEERING")
    print("-" * 80)
    
    # 3.1 Temporal features
    print("\n3.1 Creating Temporal Features")
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Year'] = df['Date'].dt.year
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    pandemic_start = df['Date'].min()
    df['Days_Since_Start'] = (df['Date'] - pandemic_start).dt.days
    
    def compute_days_since_threshold(group, threshold=100):
        first_date = group.loc[group['Confirmed'] >= threshold, 'Date'].min()
        if pd.isna(first_date):
            return pd.Series([np.nan] * len(group), index=group.index)
        return (group['Date'] - first_date).dt.days
    
    df['Days_Since_100'] = df.groupby('Country/Region', group_keys=False).apply(
        lambda g: compute_days_since_threshold(g, threshold=100)
    )
    print("✓ Temporal features created")
    
    # 3.2 Growth metrics
    print("\n3.2 Computing Growth Metrics")
    df['Growth_Rate'] = df.groupby(group_keys)['Daily_Cases'].transform(
        lambda s: safe_growth_rate(s, threshold=50)
    )
    df['Death_Growth'] = df.groupby(group_keys)['Daily_Deaths'].transform(
        lambda s: safe_growth_rate(s, threshold=10)
    )
    df['Acceleration'] = df.groupby(group_keys)['Growth_Rate'].diff()
    
    df['Doubling_Time'] = np.where(
        df['Growth_Rate'] > 0,
        np.log(2) / np.log(1 + df['Growth_Rate']),
        np.nan
    )
    df['Doubling_Time'] = df['Doubling_Time'].replace([np.inf, -np.inf], np.nan)
    
    df['Log_Cases'] = np.log1p(df['Daily_Cases'])
    df['Log_Deaths'] = np.log1p(df['Daily_Deaths'])
    print("✓ Growth metrics created")
    
    # 3.3 Severity metrics
    print("\n3.3 Computing Severity Metrics")
    df['CFR'] = np.where(df['Confirmed'] > 0, (df['Deaths'] / df['Confirmed']) * 100, 0)
    df['Active_Cases'] = (df['Confirmed'] - df['Deaths'] - df['Recovered']).clip(lower=0)
    df['Recovery_Rate'] = np.where(df['Confirmed'] > 0, df['Recovered'] / df['Confirmed'], 0)
    df['Death_to_Case_Ratio'] = np.where(df['Daily_Cases'] > 0, df['Daily_Deaths'] / df['Daily_Cases'], 0)
    print("✓ Severity metrics created")
    
    # 3.4 Intervention indicators
    print("\n3.4 Creating Intervention Indicators")
    df['NPI_Phase'] = df['Date'].apply(assign_npi_phase)
    df['Vaccine_Period'] = np.where(df['Date'] >= VACCINE_START, 'Post-vaccine', 'Pre-vaccine')
    df['Is_Lockdown'] = (df['NPI_Phase'] == 'Lockdown').astype(int)
    df['Is_Post_Vaccine'] = (df['Vaccine_Period'] == 'Post-vaccine').astype(int)
    print("✓ Intervention indicators created")
    
    print("\n[STEP 3 COMPLETE]")
    
    # ========================================================================
    # STEP 4: POPULATION NORMALIZATION
    # ========================================================================
    print("\n[STEP 4] POPULATION NORMALIZATION")
    print("-" * 80)
    
    df['Population'] = df['Country/Region'].map(POPULATION_DATA)
    missing_pop = df['Population'].isna().sum()
    if missing_pop > 0:
        median_pop = df['Population'].median()
        df['Population'] = df['Population'].fillna(median_pop)
        print(f"⚠ Filled {missing_pop:,} missing population values with median")
    
    df['Cases_per_100k'] = (df['Confirmed'] / df['Population']) * 100000
    df['Deaths_per_100k'] = (df['Deaths'] / df['Population']) * 100000
    print("✓ Population-normalized metrics created")
    
    print("\n[STEP 4 COMPLETE]")
    
    # ========================================================================
    # STEP 5: CREATE TARGET VARIABLE (7-DAY AHEAD)
    # ========================================================================
    print("\n[STEP 5] CREATING TARGET VARIABLE (7-DAY AHEAD)")
    print("-" * 80)
    
    # Create future versions of key metrics
    forecast_metrics = ['Growth_Rate', 'Cases_per_100k', 'Doubling_Time', 'CFR']
    
    print(f"\n5.1 Creating {HORIZON_DAYS}-day ahead features")
    for metric in forecast_metrics:
        if metric in df.columns:
            df[f'{metric}_future7d'] = df.groupby(group_keys)[metric].shift(-HORIZON_DAYS)
    
    # Assign warning levels
    print("\n5.2 Assigning Warning Levels (7-day ahead)")
    TARGET_COL = 'Warning_Level_7d_Ahead'
    df[TARGET_COL] = df.apply(
        lambda row: assign_warning_level(
            row['Growth_Rate_future7d'],
            row['Cases_per_100k_future7d'],
            row['Doubling_Time_future7d'],
            row['CFR_future7d']
        ),
        axis=1
    )
    
    print(f"\n✓ Created target variable: {TARGET_COL}")
    print("\nWarning level distribution:")
    print(df[TARGET_COL].value_counts())
    
    print("\n[STEP 5 COMPLETE]")
    
    # ========================================================================
    # STEP 6: SAVE PREPARED DATA
    # ========================================================================
    print("\n[STEP 6] SAVING PREPARED DATA")
    print("-" * 80)
    
    df.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    print(f"✓ Total rows: {len(df):,}")
    print(f"✓ Total columns: {len(df.columns)}")
    
    print("\n[STEP 6 COMPLETE]")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE!")
    print("="*80)
    
    print(f"""
SUMMARY:
• Dataset: {len(df):,} rows × {len(df.columns)} columns
• Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}
• Countries: {df['Country/Region'].nunique()}
• Province/State Groups: {df.groupby(group_keys).ngroups:,}

KEY FEATURE GROUPS:
• Temporal: DayOfWeek, Month, IsWeekend, Days_Since_Start, Days_Since_100
• Growth: Growth_Rate, Doubling_Time, Acceleration, Log_Cases, Death_Growth
• Severity: CFR, Active_Cases, Recovery_Rate, Death_to_Case_Ratio
• Normalized: Cases_per_100k, Deaths_per_100k
• Intervention: NPI_Phase, Vaccine_Period, Is_Lockdown, Is_Post_Vaccine
• Target: {TARGET_COL}

STATUS: READY FOR MODELING ✅
""")
    
    return df

if __name__ == '__main__':
    load_and_prepare_data()

## Project Overview

This repository contains a comprehensive data intelligence platform designed to analyze 
the evolving electric vehicle (EV) landscape. By integrating datasets from the EPA and 
NREL, the system provides stakeholders with actionable insights into vehicle efficiency, 
market segmentation, and charging infrastructure readiness.

## Key Features

- **Market Performance Analytics**: Tracking sales and market share across a total of 
  **483,360 vehicles** sold over the 2019–2025 analysis period.
- **Infrastructure Geospatial Mapping**: Visualization and analysis of **20,208 charging 
  stations** in California, including urban vs. rural distribution.
- **Vehicle Intelligence**: Deep-dive metrics on **105 unique vehicle models** with 
  dynamic efficiency benchmarking across 6 manufacturers.
- **Automated Segmentation**: K-Means clustering to categorize the market into segments 
  based on efficiency and price points (optimal k determined per run via elbow method).
- **Forecasting Engine**: Predictive modeling to project 12-month sales trends and 
  adoption rates.

## Technical Architecture

### Data Pipeline

The system utilizes an automated Python pipeline that cleans, integrates, and exports 
processed data to a centralized directory for dashboard updates:

- `outputs/processed_data/epa_vehicles_20251002.csv`
- `outputs/processed_data/charging_stations_CA_20251002.csv`

### Tech Stack

- **Frontend**: Streamlit (Interactive Web Interface)
- **Data Processing**: Pandas, NumPy, SciPy
- **Machine Learning**: Scikit-learn (K-Means Clustering, Linear Regression)
- **Visualization**: Plotly Express, Matplotlib, Seaborn, Folium

## Installation & Setup

1. **Clone the repository**:
```bash
   git clone https://github.com/evantofu/ev-performance-analysis.git
   cd ev-performance-analysis
```

2. **Install dependencies**:
```bash
   pip install -r requirements.txt
```

3. **Run the Dashboard**:
```bash
   streamlit run app.py
```

## Project Structure

- `/data/raw`: Raw datasets from EPA and NREL.
- `/outputs/processed_data`: Cleaned CSV files ready for analysis.
- `/notebooks`: Core data science and ML experimentation.
- `app.py`: Streamlit application code.

## Results Summary

The efficiency leader across the 6 tracked manufacturers is determined dynamically
from the EPA dataset. Infrastructure analysis of 20,208 California stations identifies
underserved clusters via K-Means and maps urban vs. rural distribution — with urban
classification based on city population thresholds. DC fast charging availability and
network concentration are also surfaced as key infrastructure metrics.

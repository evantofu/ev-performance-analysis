# EV Performance Analytics & Market Intelligence Platform

A comprehensive data analytics platform for electric vehicle market intelligence, performance benchmarking, and infrastructure planning using advanced machine learning and statistical modeling.

## Overview

This project provides an end-to-end analytical pipeline for electric vehicle market analysis, featuring sophisticated time series forecasting, automated market segmentation, and geospatial infrastructure analysis. The platform transforms raw EV data into actionable business intelligence through:

- **Advanced Predictive Modeling**: Time series forecasting with Facebook Prophet and regression analysis
- **Automated Market Segmentation**: K-means clustering with elbow method optimization
- **Geospatial Intelligence**: Urban vs. rural infrastructure analysis with interactive mapping
- **Executive Dashboards**: Comprehensive business intelligence with strategic recommendations

## Key Features

### 🔮 Predictive Analytics
- **Sales Forecasting**: Facebook Prophet implementation with confidence intervals and seasonal decomposition
- **Efficiency Trends**: Longitudinal analysis of MPGe improvements across manufacturers
- **Market Share Projections**: Linear regression models for market penetration forecasting
- **Infrastructure Planning**: Data-driven recommendations for charging station expansion

### 🎯 Market Intelligence
- **Competitive Benchmarking**: Manufacturer performance across efficiency, range, and pricing
- **Automated Segmentation**: K-means clustering with optimal cluster detection identifies 3 distinct market segments
- **Strategic Positioning**: Price-performance analysis and market gap identification
- **Growth Metrics**: CAGR calculations and YoY growth analysis

### 🗺️ Geospatial Analysis
- **Infrastructure Mapping**: Folium-based interactive maps of charging station distribution
- **Urban-Rural Analysis**: Geographic segmentation of charging infrastructure
- **Network Analysis**: Market share and connector type distribution across charging networks
- **Coverage Gaps**: Identification of underserved areas using clustering techniques

### 📊 Visualization & Reporting
- **Interactive Dashboards**: Comprehensive executive dashboard with performance KPIs
- **Advanced Visualizations**: Parallel coordinates, scatter matrices, and time series plots
- **Automated Reporting**: PDF/excel report generation with key insights
- **Export Functionality**: Processed data export for further analysis

## Technical Implementation

### Advanced Analytics
- **Time Series Forecasting**: Facebook Prophet with seasonal decomposition
- **Clustering**: K-means with automated elbow detection for optimal segmentation
- **Regression Analysis**: Linear regression for trend forecasting and correlations
- **Statistical Testing**: Hypothesis testing and confidence interval calculation

### Data Processing
- **Data Validation**: Comprehensive data quality checks and missing value imputation
- **Feature Engineering**: Derived metrics for efficiency, value, and market positioning
- **Normalization**: Standard scaling for clustering analysis
- **Temporal Analysis**: Time-based aggregation and trend calculation

## Installation & Setup

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd ev-performance-analysis
```

2. **Create virtual environment**:
```bash
python -m venv ev-env
source ev-env/bin/activate  # On Windows: ev-env\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install Prophet (additional requirement)**:
```bash
pip install prophet
```

5. **Set up directory structure**:
```bash
mkdir -p data/raw outputs/figures outputs/processed_data
```

## Usage

### Comprehensive Analysis Notebook

1. **Run the Jupyter notebook**:
```bash
jupyter notebook 02_ev_performance_analysis.md
```

2. **Execute the complete analysis pipeline**:
   - Data loading and validation
   - Efficiency trend analysis by manufacturer
   - Charging infrastructure analysis
   - Market growth forecasting
   - Competitive positioning analysis
   - Automated market segmentation
   - Executive dashboard generation
   - Report export

### Data Requirements

Place the following files in `data/raw/`:
- `epa_vehicles_YYYYMMDD.csv` - Vehicle performance data (MPGe, range, pricing)
- `charging_stations_CA_YYYYMMDD.csv` - Charging station data (location, network, connector types)
- `ev_sales_data_YYYYMMDD.csv` - Historical sales data with market share

## Project Structure

```
ev-performance-analysis/
├── data/
│   └── raw/                 # Raw data files (EPA, NREL, sales data)
├── outputs/
│   ├── figures/             # Generated visualizations (PNG, HTML)
│   └── processed_data/      # Processed data exports (CSV, JSON)
├── 02_ev_performance_analysis.md  # Main analysis notebook
├── requirements.txt         # Python dependencies
├── app.py                  # Streamlit dashboard (optional)
└── README.md              # This file
```

## Technical Stack

- **Python 3.8+**: Core programming language
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly, folium
- **Machine Learning**: scikit-learn, prophet, scipy
- **Dashboard**: Streamlit (optional)
- **Notebook**: Jupyter

## Key Insights Delivered

1. **Market Trends**: CAGR analysis and growth projections
2. **Technology Benchmarks**: Efficiency leaders and improvement rates
3. **Infrastructure Gaps**: Urban-rural distribution and coverage needs
4. **Competitive Landscape**: Manufacturer positioning and market segments
5. **Strategic Recommendations**: Data-driven guidance for manufacturers and investors

## Applications

- **Automotive Manufacturers**: Product positioning and R&D planning
- **Infrastructure Providers**: Charging network expansion strategy
- **Investors**: Market opportunity analysis and trend identification
- **Policy Makers**: EV adoption planning and infrastructure development
- **Researchers**: Academic study of EV market dynamics

## License

This project is developed for portfolio purposes. Ensure proper data usage rights for commercial applications.

## Acknowledgments

- Data sources: EPA Fuel Economy Data, NREL Charging Station Data
- Analysis tools: Python Data Science ecosystem, Facebook Prophet
- Visualization: Plotly, Folium, Matplotlib
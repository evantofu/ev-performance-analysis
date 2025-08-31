# EV Performance Analysis & Dashboard

A comprehensive analysis and interactive dashboard for electric vehicle market intelligence, performance metrics, and charging infrastructure analysis.

## Overview

This project provides a complete analytical pipeline for electric vehicle (EV) data, featuring:

- **Jupyter Notebook Analysis**: Comprehensive data processing, visualization, and statistical modeling of EV performance metrics
- **Streamlit Dashboard**: Interactive web application for exploring EV market trends, performance metrics, and charging infrastructure
- **Advanced Analytics**: Predictive modeling, clustering analysis, and market segmentation

## Features

### Analytical Components
- EV efficiency trends by manufacturer and year
- Charging infrastructure analysis (urban vs. rural distribution)
- Market growth forecasting and sales trend analysis
- Competitive analysis of EV manufacturers
- K-means clustering for market segmentation
- Geographic mapping of charging stations

### Dashboard Features
- Interactive filters for year, manufacturer, and date ranges
- Sales forecasting with linear regression models
- Performance metrics visualization (efficiency, range, pricing)
- Charging station mapping and network analysis
- Market segmentation visualization
- Data export functionality

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd ev-performance-analysis
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv ev-env
source ev-env/bin/activate  # On Windows: ev-env\Scripts\activate
pip install -r requirements.txt
```

3. Set up the data directory structure:
```bash
mkdir -p data/raw outputs/figures outputs/processed_data
```

## Usage

### Running the Analysis Notebook

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Open and run `02_ev_performance_analysis.md` to:
   - Process raw EV data
   - Generate visualizations and insights
   - Create predictive models
   - Export processed data for the dashboard

### Running the Streamlit Dashboard

1. Ensure you've run the analysis notebook to generate processed data
2. Launch the dashboard:
```bash
streamlit run app.py
```

3. Access the dashboard at `http://localhost:8501`

### Data Requirements

Place your data files in the `data/raw/` directory:
- `epa_vehicles_YYYYMMDD.csv` - Vehicle performance data
- `charging_stations_CA_YYYYMMDD.csv` - Charging station data
- `ev_sales_data_YYYYMMDD.csv` - Sales data

## Project Structure

```
ev-performance-analysis/
├── data/
│   └── raw/                 # Raw data files
├── outputs/
│   ├── figures/             # Generated visualizations
│   └── processed_data/      # Processed data for dashboard
├── app.py                   # Streamlit dashboard
├── 02_ev_performance_analysis.md  # Jupyter notebook
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Dependencies

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, scipy
- plotly, folium
- streamlit
- jupyter

## Data Sources

- EPA Vehicle Data
- NREL Charging Station Data
- EV Sales Data (simulated for demonstration)

## License

This project is for portfolio purposes. Please ensure you have proper permissions for any data used.

## Acknowledgments

- Data provided by EPA and NREL
- Built with Python's data science ecosystem
- Dashboard powered by Streamlit
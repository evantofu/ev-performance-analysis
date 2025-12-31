## Project Overview

This repository contains a comprehensive data intelligence platform designed to analyze the evolving electric vehicle (EV) landscape. By integrating disparate datasets from the EPA and NREL, the system provides stakeholders with actionable insights into vehicle efficiency, market segmentation, and charging infrastructure readiness.

## Key Features

* **Market Performance Analytics**: Tracking sales and market share across a total market size of **483,360 vehicles**.
* **Infrastructure Geospatial Mapping**: Visualization and analysis of **20,208 charging stations** in California, including urban vs. rural distribution (currently **22.4% urban**).
* **Vehicle Intelligence**: Deep-dive metrics on 105 unique vehicle models with an average industry efficiency of **94.6 MPGe**.
* **Automated Segmentation**: Utilization of **K-Means Clustering** to categorize the market into **7 distinct segments** based on efficiency and price points.
* **Forecasting Engine**: Predictive modeling to project 12-month sales trends and adoption rates.

## Technical Architecture

### Data Pipeline

The system utilizes an automated Python pipeline that cleans, integrates, and exports processed data to a centralized directory for real-time dashboard updates:

* `outputs/processed_data/epa_vehicles_20251002.csv`
* `outputs/processed_data/charging_stations_CA_20251002.csv`

### Tech Stack

* **Frontend**: Streamlit (Interactive Web Interface)
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-learn (K-Means Clustering, Linear Regression)
* **Visualization**: Plotly Express

## Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/evantofu/ev-performance-analysis.git
cd ev-performance-analysis

```


2. **Install dependencies**:
```bash
pip install streamlit pandas plotly scikit-learn

```


3. **Run the Dashboard**:
```bash
streamlit run app.py

```



## Project Structure

* `/data`: Raw datasets from EPA and NREL.
* `/outputs/processed_data`: Cleaned CSV files ready for analysis.
* `analysis_notebook.ipynb`: Core data science and ML experimentation.
* `app.py`: Streamlit application code.

## Results Summary

The analysis identifies **Tesla** as the current efficiency leader in the identified market segments. While infrastructure is expanding, the data reveals a heavy lean toward non-urban station placement, highlighting a significant opportunity for growth in city-center charging networks.
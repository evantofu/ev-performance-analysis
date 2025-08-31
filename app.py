# app.py - Enhanced EV Industry Dashboard
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import os
import glob

# Set page config first
st.set_page_config(page_title="EV Industry Dashboard", layout="wide", page_icon="⚡")

@st.cache_data
def load_data():
    """Load data with fallback options if processed files don't exist"""
    try:
        # Try to load processed data first
        vehicles_files = glob.glob('outputs/processed_data/epa_vehicles_*.csv')
        stations_files = glob.glob('outputs/processed_data/charging_stations_CA_*.csv')
        sales_files = glob.glob('outputs/processed_data/ev_sales_data_*.csv')
        
        if vehicles_files and stations_files and sales_files:
            # Use the latest files
            latest_vehicles = max(vehicles_files, key=os.path.getctime)
            latest_stations = max(stations_files, key=os.path.getctime)
            latest_sales = max(sales_files, key=os.path.getctime)
            
            vehicles_df = pd.read_csv(latest_vehicles)
            stations_df = pd.read_csv(latest_stations)
            sales_df = pd.read_csv(latest_sales)
            
            st.success("Loaded processed data files")
            return vehicles_df, stations_df, sales_df
        else:
            # Fall back to raw data
            st.warning("Processed data not found. Using raw data with limited functionality.")
            
            # Try to load raw data
            raw_vehicles = glob.glob('data/raw/epa_vehicles_*.csv')
            raw_stations = glob.glob('data/raw/charging_stations_*.csv')
            raw_sales = glob.glob('data/raw/ev_sales_data_*.csv')
            
            if raw_vehicles and raw_stations and raw_sales:
                latest_vehicles = max(raw_vehicles, key=os.path.getctime)
                latest_stations = max(raw_stations, key=os.path.getctime)
                latest_sales = max(raw_sales, key=os.path.getctime)
                
                vehicles_df = pd.read_csv(latest_vehicles)
                stations_df = pd.read_csv(latest_stations)
                sales_df = pd.read_csv(latest_sales)
                
                return vehicles_df, stations_df, sales_df
            else:
                st.error("No data files found. Please run the analysis notebook first.")
                return None, None, None
                
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def create_sales_forecast(sales_df):
    """Create sales forecast using linear regression"""
    if 'date' not in sales_df.columns or 'total_ev_sales' not in sales_df.columns:
        return None
    
    sales_df = sales_df.sort_values('date')
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    sales_df['month_index'] = range(len(sales_df))
    
    X = sales_df[['month_index']]
    y = sales_df['total_ev_sales']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Forecast next 12 months
    future_months = 12
    future_indices = np.arange(len(sales_df), len(sales_df) + future_months).reshape(-1, 1)
    forecast = model.predict(future_indices)
    
    # Create future dates
    last_date = sales_df['date'].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1), 
        periods=future_months, 
        freq='ME'
    )
    
    forecast_df = pd.DataFrame({'date': future_dates, 'forecast': forecast})
    return sales_df, forecast_df, model

def main():
    st.title("⚡ EV Industry Intelligence Dashboard")
    st.markdown("Comprehensive analysis of electric vehicle market trends, performance, and infrastructure")
    
    # Load data
    vehicles_df, stations_df, sales_df = load_data()
    
    if vehicles_df is None or stations_df is None or sales_df is None:
        st.error("""
        ## Data Not Available
                
        Please run the analysis notebook first to generate the processed data files:
        
        1. Open and run the Jupyter notebook (`02_ev_performance_analysis.md` or similar)
        2. Make sure the notebook exports processed data to `outputs/processed_data/`
        3. Restart this Streamlit app
        
        Alternatively, place raw data files in the `data/raw/` directory.
        """)
        return
    
    # Convert date column if it exists
    if 'date' in sales_df.columns:
        sales_df['date'] = pd.to_datetime(sales_df['date'])
        sales_df = sales_df.sort_values('date')
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Initialize filtered dataframes
    filtered_vehicles = vehicles_df.copy()
    filtered_stations = stations_df.copy()
    filtered_sales = sales_df.copy()
    
    # Year filter
    if 'year' in vehicles_df.columns:
        min_year = int(vehicles_df['year'].min())
        max_year = int(vehicles_df['year'].max())
        selected_years = st.sidebar.slider(
            "Select Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
        filtered_vehicles = filtered_vehicles[
            (filtered_vehicles['year'] >= selected_years[0]) & 
            (filtered_vehicles['year'] <= selected_years[1])
        ]
    
    # Manufacturer filter
    if 'make' in vehicles_df.columns:
        manufacturers = st.sidebar.multiselect(
            "Select Manufacturers",
            options=vehicles_df['make'].unique(),
            default=vehicles_df['make'].unique()
        )
        filtered_vehicles = filtered_vehicles[filtered_vehicles['make'].isin(manufacturers)]
    
    # Date range filter for sales
    if 'date' in sales_df.columns:
        # Convert to datetime.date objects for the slider
        min_date = sales_df['date'].min().date()
        max_date = sales_df['date'].max().date()
    
        selected_dates = st.sidebar.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date)
        )
    
        # Convert back to Timestamp for filtering
        start_date = pd.Timestamp(selected_dates[0])
        end_date = pd.Timestamp(selected_dates[1])
    
        filtered_sales = filtered_sales[
            (filtered_sales['date'] >= start_date) & 
            (filtered_sales['date'] <= end_date)
        ]
    
    # Tab layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Market Overview", "Vehicle Performance", "Infrastructure", 
        "Advanced Analytics", "Data Export"
    ])
    
    with tab1:
        st.header("Market Trends & Sales")
        
        if 'total_ev_sales' in filtered_sales.columns:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_sales = filtered_sales['total_ev_sales'].sum()
                st.metric("Total EV Sales", f"{total_sales:,.0f}")
            
            with col2:
                current_sales = filtered_sales['total_ev_sales'].iloc[-1] if len(filtered_sales) > 0 else 0
                st.metric("Current Monthly Sales", f"{current_sales:,.0f}")
            
            with col3:
                if 'year' in filtered_sales.columns and len(filtered_sales) > 1:
                    yearly_sales = filtered_sales.groupby('year')['total_ev_sales'].sum()
                    if len(yearly_sales) > 1:
                        growth_rate = yearly_sales.pct_change().mean() * 100
                        st.metric("Avg Annual Growth", f"{growth_rate:.1f}%")
                    else:
                        st.metric("Avg Annual Growth", "N/A")
                else:
                    st.metric("Avg Annual Growth", "N/A")
            
            with col4:
                if 'market_share_percent' in filtered_sales.columns:
                    current_share = filtered_sales['market_share_percent'].iloc[-1]
                    st.metric("Current Market Share", f"{current_share:.1f}%")
                else:
                    st.metric("Market Share", "N/A")
            
            # Sales trend chart
            if 'date' in filtered_sales.columns and 'total_ev_sales' in filtered_sales.columns:
                fig = px.line(
                    filtered_sales, 
                    x='date', 
                    y='total_ev_sales', 
                    title='Monthly EV Sales Trend',
                    labels={'total_ev_sales': 'Units Sold', 'date': 'Date'}
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Sales forecast
            st.subheader("12-Month Sales Forecast")
            sales_forecast_data = create_sales_forecast(sales_df)  # Use full dataset for forecast
            
            if sales_forecast_data:
                historical_df, forecast_df, model = sales_forecast_data
                
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(
                    x=historical_df['date'], 
                    y=historical_df['total_ev_sales'], 
                    name='Historical',
                    line=dict(color='#1f77b4')
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_df['date'], 
                    y=forecast_df['forecast'], 
                    name='Forecast', 
                    line=dict(color='#ff7f0e', dash='dash')
                ))
                fig_forecast.update_layout(
                    title='Sales Forecast (Next 12 Months)',
                    xaxis_title='Date',
                    yaxis_title='Units Sold'
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Forecast metrics
                forecast_total = forecast_df['forecast'].sum()
                current_total = historical_df['total_ev_sales'].sum()
                growth_pct = ((forecast_total / current_total * len(historical_df) / 12) - 1) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Projected Annual Sales", f"{forecast_total:,.0f}")
                with col2:
                    st.metric("Projected Growth Rate", f"{growth_pct:.1f}%")
        
        # Market share by manufacturer
        if 'make' in filtered_vehicles.columns:
            st.subheader("Market Share by Manufacturer")
            make_counts = filtered_vehicles['make'].value_counts()
            fig_pie = px.pie(
                values=make_counts.values, 
                names=make_counts.index, 
                title='Manufacturer Model Distribution'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        st.header("Vehicle Performance Analysis")
        
        efficiency_available = 'combined_mpg' in filtered_vehicles.columns
        range_available = 'range_miles' in filtered_vehicles.columns
        price_available = 'msrp_base' in filtered_vehicles.columns
        
        if efficiency_available and range_available:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_efficiency = filtered_vehicles['combined_mpg'].mean()
                st.metric("Average Efficiency", f"{avg_efficiency:.1f} MPGe")
            
            with col2:
                avg_range = filtered_vehicles['range_miles'].mean()
                st.metric("Average Range", f"{avg_range:.1f} miles")
            
            with col3:
                if price_available:
                    avg_price = filtered_vehicles['msrp_base'].mean()
                    st.metric("Average Price", f"${avg_price:,.0f}")
                else:
                    st.metric("Average Price", "N/A")
            
            with col4:
                if 'battery_capacity_kwh' in filtered_vehicles.columns:
                    avg_battery = filtered_vehicles['battery_capacity_kwh'].mean()
                    st.metric("Avg Battery", f"{avg_battery:.1f} kWh")
                else:
                    st.metric("Avg Battery", "N/A")
            
            # Efficiency by manufacturer
            if 'make' in filtered_vehicles.columns:
                st.subheader("Efficiency by Manufacturer")
                efficiency_data = filtered_vehicles.groupby('make')['combined_mpg'].mean().sort_values(ascending=False)
                
                fig_eff = px.bar(
                    x=efficiency_data.index,
                    y=efficiency_data.values,
                    title='Average Efficiency by Manufacturer',
                    labels={'x': 'Manufacturer', 'y': 'Combined MPGe'}
                )
                fig_eff.update_xaxes(tickangle=45)
                st.plotly_chart(fig_eff, use_container_width=True)
            
            # Scatter plot: Efficiency vs Range
            st.subheader("Efficiency vs Range")
            color_by = 'make' if 'make' in filtered_vehicles.columns else None
            size_by = 'msrp_base' if price_available else None
            
            fig_scatter = px.scatter(
                filtered_vehicles, 
                x='combined_mpg', 
                y='range_miles', 
                color=color_by,
                size=size_by,
                title='Efficiency vs Range by Manufacturer',
                hover_data=['model'] if 'model' in filtered_vehicles.columns else None
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Price vs Efficiency
            if price_available:
                st.subheader("Price vs Efficiency")
                fig_price = px.scatter(
                    filtered_vehicles, 
                    x='msrp_base', 
                    y='combined_mpg', 
                    color=color_by,
                    title='Price vs Efficiency',
                    labels={'msrp_base': 'Price ($)', 'combined_mpg': 'Efficiency (MPGe)'}
                )
                st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.warning("Vehicle performance data is missing required columns")
    
    with tab3:
        st.header("Charging Infrastructure")
        
        stations_available = len(filtered_stations) > 0
        location_available = 'latitude' in filtered_stations.columns and 'longitude' in filtered_stations.columns
        
        if stations_available:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_stations = len(filtered_stations)
                st.metric("Total Stations", f"{total_stations:,}")
            
            with col2:
                if 'dc_fast_count' in filtered_stations.columns:
                    dc_fast_stations = (filtered_stations['dc_fast_count'] > 0).sum()
                    st.metric("DC Fast Stations", f"{dc_fast_stations:,}")
                else:
                    st.metric("DC Fast Stations", "N/A")
            
            with col3:
                if 'level2_count' in filtered_stations.columns:
                    level2_connectors = filtered_stations['level2_count'].sum()
                    st.metric("Level 2 Connectors", f"{level2_connectors:,}")
                else:
                    st.metric("Level 2 Connectors", "N/A")
            
            with col4:
                # Simplified urban calculation
                if 'city' in filtered_stations.columns:
                    major_cities = ['Los Angeles', 'San Diego', 'San Jose', 'San Francisco', 
                                  'Fresno', 'Sacramento', 'Long Beach', 'Oakland']
                    urban_stations = filtered_stations[filtered_stations['city'].isin(major_cities)].shape[0]
                    urban_pct = (urban_stations / len(filtered_stations)) * 100
                    st.metric("Urban Stations", f"{urban_pct:.1f}%")
                else:
                    st.metric("Urban Stations", "N/A")
            
            # Station map
            if location_available:
                st.subheader("Charging Station Map")
                sample_stations = filtered_stations.sample(n=min(1000, len(filtered_stations)))
                st.map(sample_stations[['latitude', 'longitude']])
            else:
                st.warning("Station location data not available for mapping")
            
            # Network market share
            if 'network' in filtered_stations.columns:
                st.subheader("Charging Network Market Share")
                network_counts = filtered_stations['network'].value_counts().head(10)
                
                fig_network = px.bar(
                    x=network_counts.index,
                    y=network_counts.values,
                    title='Top Charging Networks',
                    labels={'x': 'Network', 'y': 'Number of Stations'}
                )
                fig_network.update_xaxes(tickangle=45)
                st.plotly_chart(fig_network, use_container_width=True)
            
            # Connector type distribution
            connector_cols = ['level1_count', 'level2_count', 'dc_fast_count']
            available_connectors = [col for col in connector_cols if col in filtered_stations.columns]
            
            if available_connectors:
                st.subheader("Connector Type Distribution")
                connector_totals = filtered_stations[available_connectors].sum()
                
                fig_connector = px.pie(
                    values=connector_totals.values,
                    names=connector_totals.index,
                    title='Charging Connector Types'
                )
                st.plotly_chart(fig_connector, use_container_width=True)
        else:
            st.warning("Station data not available")
    
    with tab4:
        st.header("Advanced Analytics")
        
        # Clustering analysis
        st.subheader("Market Segmentation (K-means Clustering)")
        
        feature_cols = ['combined_mpg', 'range_miles', 'msrp_base']
        available_features = [col for col in feature_cols if col in filtered_vehicles.columns]
        
        if len(available_features) >= 2:
            X = filtered_vehicles[available_features].fillna(filtered_vehicles[available_features].mean())
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            filtered_vehicles['cluster'] = cluster_labels
            
            # Plot clusters
            fig_cluster = px.scatter(
                filtered_vehicles, 
                x=available_features[0], 
                y=available_features[1], 
                color='cluster',
                title=f'Vehicle Clusters: {available_features[0]} vs {available_features[1]}',
                hover_data=['make', 'model'] if 'make' in filtered_vehicles.columns and 'model' in filtered_vehicles.columns else None
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
            
            # Cluster descriptions
            st.subheader("Cluster Characteristics")
            cluster_data = []
            
            for cluster_id in sorted(filtered_vehicles['cluster'].unique()):
                cluster_df = filtered_vehicles[filtered_vehicles['cluster'] == cluster_id]
                cluster_info = {
                    'Cluster': cluster_id,
                    'Count': len(cluster_df),
                    'Primary Make': cluster_df['make'].mode().iloc[0] if 'make' in cluster_df.columns else 'N/A'
                }
                
                if 'msrp_base' in available_features:
                    cluster_info['Avg Price'] = f"${cluster_df['msrp_base'].mean():,.0f}"
                if 'combined_mpg' in available_features:
                    cluster_info['Avg Efficiency'] = f"{cluster_df['combined_mpg'].mean():.1f} MPGe"
                if 'range_miles' in available_features:
                    cluster_info['Avg Range'] = f"{cluster_df['range_miles'].mean():.1f} miles"
                
                cluster_data.append(cluster_info)
            
            cluster_df = pd.DataFrame(cluster_data)
            st.dataframe(cluster_df, use_container_width=True)
        else:
            st.warning("Insufficient data for clustering analysis")
        
        # Efficiency trends over time
        if 'year' in filtered_vehicles.columns and 'combined_mpg' in filtered_vehicles.columns:
            st.subheader("Efficiency Improvement Trends")
            yearly_efficiency = filtered_vehicles.groupby('year')['combined_mpg'].mean()
            
            fig_trend = px.line(
                x=yearly_efficiency.index,
                y=yearly_efficiency.values,
                title='Average Efficiency Improvement Over Time',
                labels={'x': 'Year', 'y': 'Average MPGe'}
            )
            fig_trend.add_scatter(
                x=yearly_efficiency.index, 
                y=yearly_efficiency.values, 
                mode='markers'
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Calculate improvement rate
            if len(yearly_efficiency) > 1:
                improvement_rate = (yearly_efficiency.iloc[-1] - yearly_efficiency.iloc[0]) / (yearly_efficiency.index.max() - yearly_efficiency.index.min())
                st.metric("Average Annual Improvement", f"{improvement_rate:.1f} MPGe/year")
    
    with tab5:
        st.header("Data Export")
        
        st.info("""
        Download the processed data used in this analysis. The data includes:
        - Vehicle performance metrics
        - Charging station information
        - Historical sales data
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Download Vehicle Data as CSV"):
                csv = filtered_vehicles.to_csv(index=False)
                st.download_button(
                    label="Download Vehicles CSV",
                    data=csv,
                    file_name="ev_vehicles_data.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Download Station Data as CSV"):
                csv = filtered_stations.to_csv(index=False)
                st.download_button(
                    label="Download Stations CSV",
                    data=csv,
                    file_name="ev_charging_stations.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("Download Sales Data as CSV"):
                csv = filtered_sales.to_csv(index=False)
                st.download_button(
                    label="Download Sales CSV",
                    data=csv,
                    file_name="ev_sales_data.csv",
                    mime="text/csv"
                )
        
        # Data summary
        st.subheader("Data Summary")
        
        summary_data = {
            "Dataset": ["Vehicles", "Charging Stations", "Sales"],
            "Records": [len(filtered_vehicles), len(filtered_stations), len(filtered_sales)],
            "Time Period": [
                f"{filtered_vehicles['year'].min()}-{filtered_vehicles['year'].max()}" if 'year' in filtered_vehicles.columns else "N/A",
                "N/A",
                f"{filtered_sales['date'].min().strftime('%Y-%m')} to {filtered_sales['date'].max().strftime('%Y-%m')}" if 'date' in filtered_sales.columns else "N/A"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**Data Sources:** EPA Vehicle Data, NREL Charging Station Data, EV Sales Data")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.markdown("**Note:** This is a demonstration dashboard for portfolio purposes")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Improved EV Performance Analysis Data Collection
Generates robust sample data for analysis
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EVDataCollector:
    def __init__(self):
        self.data_dir = Path('data/raw')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d')
        
    def create_epa_vehicles_data(self):
        """Create comprehensive EPA vehicle efficiency data"""
        logger.info("Creating EPA vehicles dataset...")
        
        # Define realistic EV models with actual-like specifications
        ev_models = [
            {
                'make': 'Tesla', 'model': 'Model 3', 'drive_type': 'RWD',
                'base_range': 272, 'base_efficiency': 132, 'battery_size': 60
            },
            {
                'make': 'Tesla', 'model': 'Model Y', 'drive_type': 'AWD',
                'base_range': 326, 'base_efficiency': 121, 'battery_size': 75
            },
            {
                'make': 'Ford', 'model': 'Mustang Mach-E', 'drive_type': 'RWD',
                'base_range': 312, 'base_efficiency': 105, 'battery_size': 88
            },
            {
                'make': 'Chevrolet', 'model': 'Bolt EUV', 'drive_type': 'FWD',
                'base_range': 247, 'base_efficiency': 120, 'battery_size': 65
            },
            {
                'make': 'BMW', 'model': 'iX', 'drive_type': 'AWD',
                'base_range': 324, 'base_efficiency': 86, 'battery_size': 105.2
            },
            {
                'make': 'Hyundai', 'model': 'Ioniq 5', 'drive_type': 'RWD',
                'base_range': 305, 'base_efficiency': 114, 'battery_size': 77.4
            },
            {
                'make': 'Rivian', 'model': 'R1T', 'drive_type': 'AWD',
                'base_range': 314, 'base_efficiency': 70, 'battery_size': 135
            },
            {
                'make': 'Rivian', 'model': 'R1S', 'drive_type': 'AWD',
                'base_range': 316, 'base_efficiency': 69, 'battery_size': 135
            }
        ]
        
        years = range(2019, 2025)
        vehicles_data = []
        
        for model_info in ev_models:
            for year in years:
                # Simulate year-over-year improvements
                year_factor = 1 + (year - 2019) * 0.03  # 3% improvement per year
                
                # Add some realistic variation
                range_variation = np.random.normal(1, 0.05)
                efficiency_variation = np.random.normal(1, 0.03)
                
                record = {
                    'year': year,
                    'make': model_info['make'],
                    'model': model_info['model'],
                    'drive_type': model_info['drive_type'],
                    'fuel_type': 'Electric',
                    'vehicle_class': 'Midsize Cars' if 'Model 3' in model_info['model'] else 'Small SUV',
                    'engine_description': 'Electric Motor',
                    'transmission': 'Automatic (variable gear ratios)',
                    'city_mpg': round(model_info['base_efficiency'] * year_factor * efficiency_variation),
                    'highway_mpg': round(model_info['base_efficiency'] * year_factor * efficiency_variation * 0.9),
                    'combined_mpg': round(model_info['base_efficiency'] * year_factor * efficiency_variation * 0.95),
                    'range_miles': round(model_info['base_range'] * year_factor * range_variation),
                    'battery_capacity_kwh': model_info['battery_size'],
                    'charge_time_240v': round(model_info['battery_size'] / 7.2, 1),  # Assuming 7.2kW home charging
                    'msrp_base': round(35000 + np.random.normal(15000, 5000)),  # Realistic pricing
                    'co2_emissions': 0,
                    'ghg_score': 10
                }
                vehicles_data.append(record)
        
        vehicles_df = pd.DataFrame(vehicles_data)
        
        # Save to CSV
        filename = f'epa_vehicles_{self.timestamp}.csv'
        filepath = self.data_dir / filename
        vehicles_df.to_csv(filepath, index=False)
        logger.info(f"✅ Created {filename} with {len(vehicles_df)} records")
        
        return vehicles_df
    
    def get_charging_stations_data(self):
        """Get real NREL charging station data for California"""
        logger.info("Fetching NREL charging stations data...")
        
        try:
            # NREL Alternative Fuel Data Center API
            url = "https://developer.nrel.gov/api/alt-fuel-stations/v1.json"
            params = {
                'fuel_type': 'ELEC',
                'state': 'CA',
                'limit': 'all',
                'format': 'json'
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                stations = data.get('fuel_stations', [])
                
                if stations:
                    # Process the real data
                    stations_data = []
                    for station in stations[:20000]:  # Limit to avoid memory issues
                        stations_data.append({
                            'station_name': station.get('station_name', 'Unknown'),
                            'street_address': station.get('street_address', ''),
                            'city': station.get('city', ''),
                            'state': station.get('state', ''),
                            'zip_code': station.get('zip', ''),
                            'latitude': station.get('latitude', 0),
                            'longitude': station.get('longitude', 0),
                            'access_code': station.get('access_code', 'Unknown'),
                            'facility_type': station.get('facility_type', 'Unknown'),
                            'network': station.get('ev_network', 'Unknown'),
                            'connector_types': station.get('ev_connector_types', ''),
                            'level1_count': station.get('ev_level1_evse_num', 0) or 0,
                            'level2_count': station.get('ev_level2_evse_num', 0) or 0,
                            'dc_fast_count': station.get('ev_dc_fast_num', 0) or 0,
                            'pricing': station.get('ev_pricing', 'Unknown'),
                            'hours': station.get('access_days_time', 'Unknown'),
                            'date_last_confirmed': station.get('date_last_confirmed', ''),
                            'updated_at': station.get('updated_at', '')
                        })
                    
                    stations_df = pd.DataFrame(stations_data)
                    filename = f'charging_stations_CA_{self.timestamp}.csv'
                    filepath = self.data_dir / filename
                    stations_df.to_csv(filepath, index=False)
                    logger.info(f"✅ Created {filename} with {len(stations_df)} real charging stations")
                    return stations_df
            
        except Exception as e:
            logger.warning(f"NREL API error: {e}. Creating sample data instead.")
        
        # Fallback: Create realistic sample charging station data
        return self.create_sample_charging_stations()
    
    def create_sample_charging_stations(self):
        """Create realistic sample charging station data for California"""
        logger.info("Creating sample charging stations data...")
        
        # California major cities with approximate coordinates
        ca_cities = [
            {'city': 'Los Angeles', 'lat': 34.0522, 'lon': -118.2437, 'weight': 0.3},
            {'city': 'San Francisco', 'lat': 37.7749, 'lon': -122.4194, 'weight': 0.15},
            {'city': 'San Diego', 'lat': 32.7157, 'lon': -117.1611, 'weight': 0.12},
            {'city': 'San Jose', 'lat': 37.3382, 'lon': -121.8863, 'weight': 0.08},
            {'city': 'Sacramento', 'lat': 38.5816, 'lon': -121.4944, 'weight': 0.06},
            {'city': 'Oakland', 'lat': 37.8044, 'lon': -122.2712, 'weight': 0.05},
            {'city': 'Fresno', 'lat': 36.7378, 'lon': -119.7871, 'weight': 0.04},
            {'city': 'Santa Barbara', 'lat': 34.4208, 'lon': -119.6982, 'weight': 0.03},
        ]
        
        networks = ['ChargePoint', 'Electrify America', 'EVgo', 'Tesla Supercharger', 'Blink', 'SemaConnect', 'Volta']
        facility_types = ['Shopping Center', 'Hotel', 'Gas Station', 'Workplace', 'Public', 'Multi-unit Dwelling']
        
        stations_data = []
        total_stations = 15000
        
        for i in range(total_stations):
            # Choose city based on population weight
            city = np.random.choice(ca_cities, p=[c['weight'] for c in ca_cities])
            
            # Add some geographic spread around the city center
            lat_offset = np.random.normal(0, 0.1)  # ~11 km standard deviation
            lon_offset = np.random.normal(0, 0.1)
            
            # Generate realistic station data
            network = np.random.choice(networks)
            facility_type = np.random.choice(facility_types)
            
            # Different networks have different typical configurations
            if network == 'Tesla Supercharger':
                level2_count = 0
                dc_fast_count = np.random.randint(4, 20)
                level1_count = 0
            elif network == 'Electrify America':
                level2_count = 0
                dc_fast_count = np.random.randint(2, 8)
                level1_count = 0
            else:
                level1_count = np.random.poisson(0.5)
                level2_count = np.random.poisson(3)
                dc_fast_count = np.random.poisson(0.8)
            
            station = {
                'station_name': f"{network} Station {i+1}",
                'street_address': f"{np.random.randint(100, 9999)} Main St",
                'city': city['city'],
                'state': 'CA',
                'zip_code': f"9{np.random.randint(0, 5)}{np.random.randint(100, 999):03d}",
                'latitude': round(city['lat'] + lat_offset, 4),
                'longitude': round(city['lon'] + lon_offset, 4),
                'access_code': np.random.choice(['Public', 'Private'], p=[0.8, 0.2]),
                'facility_type': facility_type,
                'network': network,
                'connector_types': 'J1772, CCS',
                'level1_count': level1_count,
                'level2_count': level2_count,
                'dc_fast_count': dc_fast_count,
                'pricing': np.random.choice(['$0.30/kWh', '$0.35/kWh', '$0.40/kWh', 'Free', 'Membership required']),
                'hours': np.random.choice(['24/7', 'Business hours', 'Dawn to dusk']),
                'date_last_confirmed': (datetime.now() - timedelta(days=np.random.randint(1, 365))).strftime('%Y-%m-%d'),
                'updated_at': datetime.now().strftime('%Y-%m-%d')
            }
            stations_data.append(station)
        
        stations_df = pd.DataFrame(stations_data)
        filename = f'charging_stations_CA_{self.timestamp}.csv'
        filepath = self.data_dir / filename
        stations_df.to_csv(filepath, index=False)
        logger.info(f"✅ Created {filename} with {len(stations_df)} sample charging stations")
        
        return stations_df
    
    def create_ev_sales_data(self):
        """Create realistic EV sales trend data"""
        logger.info("Creating EV sales trend data...")
        
        # Generate monthly data from Jan 2019 to present
        start_date = datetime(2019, 1, 1)
        end_date = datetime.now()
        
        sales_data = []
        current_date = start_date
        
        # Base monthly sales with realistic growth trend
        base_sales = 5000  # Starting monthly EV sales
        
        while current_date <= end_date:
            # Calculate months since start for trend
            months_elapsed = (current_date.year - 2019) * 12 + (current_date.month - 1)
            
            # Exponential growth with some seasonality
            growth_factor = 1.05 ** (months_elapsed / 12)  # 5% annual growth rate
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * current_date.month / 12)  # Seasonal variation
            
            # COVID impact (reduced sales in 2020)
            covid_factor = 1.0
            if current_date.year == 2020:
                covid_factor = 0.7 + 0.3 * (current_date.month / 12)  # Recovery throughout 2020
            
            # Random monthly variation
            random_factor = np.random.normal(1, 0.1)
            
            monthly_sales = round(base_sales * growth_factor * seasonal_factor * covid_factor * random_factor)
            
            sales_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'year': current_date.year,
                'month': current_date.month,
                'month_name': current_date.strftime('%B'),
                'quarter': f"Q{(current_date.month-1)//3 + 1}",
                'total_ev_sales': monthly_sales,
                'tesla_sales': round(monthly_sales * np.random.uniform(0.5, 0.7)),  # Tesla market share
                'other_premium_sales': round(monthly_sales * np.random.uniform(0.1, 0.2)),
                'mass_market_sales': round(monthly_sales * np.random.uniform(0.1, 0.3)),
                'market_share_percent': round(np.random.uniform(2, 8), 1) if current_date.year >= 2020 else round(np.random.uniform(1, 3), 1),
                'avg_price': round(35000 + np.random.normal(15000, 5000)),
                'incentives_total': round(np.random.uniform(5000, 12000))
            })
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        sales_df = pd.DataFrame(sales_data)
        filename = f'ev_sales_data_{self.timestamp}.csv'
        filepath = self.data_dir / filename
        sales_df.to_csv(filepath, index=False)
        logger.info(f"✅ Created {filename} with {len(sales_df)} monthly records")
        
        return sales_df
    
    def create_summary_file(self, vehicles_df, stations_df, sales_df):
        """Create a summary JSON file with dataset information"""
        summary = {
            'generation_date': datetime.now().isoformat(),
            'datasets': {
                'epa_vehicles': {
                    'records': len(vehicles_df),
                    'years_covered': f"{vehicles_df['year'].min()}-{vehicles_df['year'].max()}",
                    'unique_models': len(vehicles_df.groupby(['make', 'model'])),
                    'columns': list(vehicles_df.columns)
                },
                'charging_stations': {
                    'records': len(stations_df),
                    'unique_cities': len(stations_df['city'].unique()),
                    'total_connectors': {
                        'level1': int(stations_df['level1_count'].sum()),
                        'level2': int(stations_df['level2_count'].sum()),
                        'dc_fast': int(stations_df['dc_fast_count'].sum())
                    },
                    'columns': list(stations_df.columns)
                },
                'ev_sales': {
                    'records': len(sales_df),
                    'date_range': f"{sales_df['date'].min()} to {sales_df['date'].max()}",
                    'total_sales': int(sales_df['total_ev_sales'].sum()),
                    'columns': list(sales_df.columns)
                }
            }
        }
        
        filename = f'data_summary_{self.timestamp}.json'
        filepath = self.data_dir / filename
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"✅ Created summary file: {filename}")
        return summary
    
    def collect_all_data(self):
        """Run the complete data collection process"""
        logger.info("Starting EV data collection...")
        
        try:
            # Create all datasets
            vehicles_df = self.create_epa_vehicles_data()
            stations_df = self.get_charging_stations_data()
            sales_df = self.create_ev_sales_data()
            
            # Create summary
            summary = self.create_summary_file(vehicles_df, stations_df, sales_df)
            
            logger.info("✅ Data collection completed successfully!")
            logger.info(f"Generated files in {self.data_dir}:")
            for file in self.data_dir.glob('*.csv'):
                logger.info(f"  - {file.name}")
            for file in self.data_dir.glob('*.json'):
                logger.info(f"  - {file.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Data collection failed: {e}")
            return False

if __name__ == "__main__":
    collector = EVDataCollector()
    success = collector.collect_all_data()
    
    if success:
        print("\n🎉 Data collection completed! You can now run your Jupyter notebook analysis.")
    else:
        print("\n❌ Data collection failed. Check the logs above for details.")
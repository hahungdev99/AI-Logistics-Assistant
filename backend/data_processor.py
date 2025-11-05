"""
Data Processor for Hermes AI Logistics Assistant
Handles all data loading and preprocessing operations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class DataProcessor:
    def __init__(self, data_path: str = "data/shipments.csv"):
        """Initialize the data processor with shipment data"""
        self.data_path = data_path
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load and preprocess the shipment data"""
        try:
            self.df = pd.read_csv(self.data_path)
            # Convert date column to datetime
            self.df['date'] = pd.to_datetime(self.df['date'])
            # Add derived columns
            self.df['is_delayed'] = self.df['delay_minutes'] > 0
            self.df['week'] = self.df['date'].dt.isocalendar().week
            self.df['month'] = self.df['date'].dt.month
            self.df['day_of_week'] = self.df['date'].dt.dayofweek
            print(f"✓ Loaded {len(self.df)} shipment records")
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise
    
    def get_data(self) -> pd.DataFrame:
        """Return the full dataframe"""
        return self.df.copy()
    
    def get_date_range(self) -> Tuple[datetime, datetime]:
        """Get the date range of the dataset"""
        return self.df['date'].min(), self.df['date'].max()
    
    def filter_by_date_range(self, start_date: Optional[str] = None, 
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """Filter data by date range"""
        df = self.df.copy()
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        return df
    
    def filter_by_period(self, period: str) -> pd.DataFrame:
        """
        Filter data by common period names
        period: 'last week', 'this week', 'last month', 'this month', 'today', 'yesterday'
        """
        today = self.df['date'].max()
        
        if period == "today":
            return self.df[self.df['date'] == today]
        
        elif period == "yesterday":
            yesterday = today - timedelta(days=1)
            return self.df[self.df['date'] == yesterday]
        
        elif period == "last week":
            week_ago = today - timedelta(days=7)
            return self.df[(self.df['date'] > week_ago) & (self.df['date'] <= today)]
        
        elif period == "this week":
            # Get current week
            current_week = today.isocalendar()[1]
            return self.df[self.df['week'] == current_week]
        
        elif period == "last month":
            last_month = (today.replace(day=1) - timedelta(days=1)).month
            return self.df[self.df['month'] == last_month]
        
        elif period == "this month":
            current_month = today.month
            return self.df[self.df['month'] == current_month]
        
        else:
            return self.df
    
    def get_summary_stats(self) -> Dict:
        """Get overall summary statistics"""
        total_shipments = len(self.df)
        delayed_shipments = self.df['is_delayed'].sum()
        on_time_rate = (total_shipments - delayed_shipments) / total_shipments * 100
        avg_delay = self.df[self.df['is_delayed']]['delay_minutes'].mean()
        
        return {
            "total_shipments": int(total_shipments),
            "delayed_shipments": int(delayed_shipments),
            "on_time_shipments": int(total_shipments - delayed_shipments),
            "on_time_rate_percent": round(on_time_rate, 2),
            "average_delay_minutes": round(avg_delay, 2) if not np.isnan(avg_delay) else 0,
            "date_range": f"{self.df['date'].min().date()} to {self.df['date'].max().date()}"
        }
    
    def get_delay_by_reason(self) -> pd.DataFrame:
        """Get delays grouped by reason"""
        delayed = self.df[self.df['is_delayed']]
        if len(delayed) == 0:
            return pd.DataFrame(columns=['delay_reason', 'count', 'avg_delay_minutes'])
        
        summary = delayed.groupby('delay_reason').agg({
            'id': 'count',
            'delay_minutes': 'mean'
        }).reset_index()
        summary.columns = ['delay_reason', 'count', 'avg_delay_minutes']
        summary['avg_delay_minutes'] = summary['avg_delay_minutes'].round(2)
        return summary.sort_values('count', ascending=False)
    
    def get_warehouse_performance(self) -> pd.DataFrame:
        """Get warehouse performance metrics"""
        summary = self.df.groupby('warehouse').agg({
            'id': 'count',
            'processing_time_hours': 'mean',
            'delay_minutes': 'mean',
            'is_delayed': 'sum',
            'delivery_time_days': 'mean'
        }).reset_index()
        
        summary.columns = ['warehouse', 'total_shipments', 'avg_processing_time_hours', 
                          'avg_delay_minutes', 'delayed_count', 'avg_delivery_time_days']
        
        summary['on_time_rate_percent'] = (
            (summary['total_shipments'] - summary['delayed_count']) / summary['total_shipments'] * 100
        ).round(2)
        
        # Round numerical columns
        for col in ['avg_processing_time_hours', 'avg_delay_minutes', 'avg_delivery_time_days']:
            summary[col] = summary[col].round(2)
        
        return summary
    
    def get_route_performance(self) -> pd.DataFrame:
        """Get route performance metrics"""
        summary = self.df.groupby('route').agg({
            'id': 'count',
            'delay_minutes': ['mean', 'sum'],
            'is_delayed': 'sum',
            'delivery_time_days': 'mean'
        }).reset_index()
        
        summary.columns = ['route', 'total_shipments', 'avg_delay_minutes', 
                          'total_delay_minutes', 'delayed_count', 'avg_delivery_time_days']
        
        summary['on_time_rate_percent'] = (
            (summary['total_shipments'] - summary['delayed_count']) / summary['total_shipments'] * 100
        ).round(2)
        
        # Round numerical columns
        for col in ['avg_delay_minutes', 'total_delay_minutes', 'avg_delivery_time_days']:
            summary[col] = summary[col].round(2)
        
        return summary
    
    def get_time_series_data(self, metric: str = 'delay_minutes', 
                            aggregation: str = 'mean') -> pd.DataFrame:
        """Get time series data for a specific metric"""
        if aggregation == 'mean':
            ts = self.df.groupby('date')[metric].mean().reset_index()
        elif aggregation == 'sum':
            ts = self.df.groupby('date')[metric].sum().reset_index()
        elif aggregation == 'count':
            ts = self.df.groupby('date')[metric].count().reset_index()
        
        ts.columns = ['date', metric]
        if metric in ['delay_minutes', 'processing_time_hours', 'delivery_time_days']:
            ts[metric] = ts[metric].round(2)
        
        return ts
    
    def prepare_prediction_data(self, target: str = 'delay_minutes') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for prediction models
        Returns features (X) and target (y)
        """
        # Aggregate by date
        daily = self.df.groupby('date').agg({
            target: 'mean',
            'is_delayed': 'sum',
            'id': 'count'
        }).reset_index()
        
        daily['days_since_start'] = (daily['date'] - daily['date'].min()).dt.days
        
        X = daily[['days_since_start']].values
        y = daily[target].values
        
        return X, y
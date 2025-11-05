"""
Tools for Hermes AI Logistics Assistant
Contains all the tools that the LangChain agent can use
"""

from langchain.tools import tool
from typing import Optional, List, Dict
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import pandas as pd

# Global data processor instance (will be set by agent.py)
data_processor = None

def set_data_processor(processor):
    """Set the global data processor instance"""
    global data_processor
    data_processor = processor

@tool
def get_overall_statistics() -> str:
    """
    Get overall logistics statistics including total shipments, delays, and on-time rates.
    Use this tool when the user asks for general statistics or overview.
    
    Returns:
        JSON string with overall statistics
    """
    stats = data_processor.get_summary_stats()
    return json.dumps(stats, indent=2)

@tool
def get_most_delayed_route(period: str = "all") -> str:
    """
    Find the route with the most delays in a given period.
    
    Args:
        period: Time period to analyze. Options: 'last week', 'this week', 'last month', 
                'this month', 'today', 'yesterday', or 'all' for all data
    
    Returns:
        JSON string with route delay information
    """
    if period != "all":
        df = data_processor.filter_by_period(period)
    else:
        df = data_processor.get_data()
    
    if len(df) == 0:
        return json.dumps({"error": "No data found for the specified period"})
    
    route_delays = df.groupby('route').agg({
        'is_delayed': 'sum',
        'delay_minutes': ['mean', 'sum'],
        'id': 'count'
    }).reset_index()
    
    route_delays.columns = ['route', 'delayed_count', 'avg_delay', 'total_delay', 'total_shipments']
    route_delays = route_delays.sort_values('delayed_count', ascending=False)
    
    most_delayed = route_delays.iloc[0]
    
    result = {
        "period": period,
        "most_delayed_route": most_delayed['route'],
        "delayed_shipments": int(most_delayed['delayed_count']),
        "total_shipments": int(most_delayed['total_shipments']),
        "average_delay_minutes": round(float(most_delayed['avg_delay']), 2),
        "total_delay_minutes": round(float(most_delayed['total_delay']), 2),
        "all_routes_ranking": route_delays.to_dict('records')
    }
    
    return json.dumps(result, indent=2)

@tool
def get_top_warehouses_by_processing_time(k: int = 3, order: str = "highest") -> str:
    """
    Get top K warehouses ranked by average processing time.
    
    Args:
        k: Number of top warehouses to return (default: 3)
        order: 'highest' for longest processing times or 'lowest' for shortest (default: 'highest')
    
    Returns:
        JSON string with warehouse rankings
    """
    warehouse_stats = data_processor.get_warehouse_performance()
    
    ascending = order == "lowest"
    top_warehouses = warehouse_stats.sort_values('avg_processing_time_hours', ascending=ascending).head(k)
    
    result = {
        "ranking_criteria": f"Top {k} warehouses by {order} processing time",
        "warehouses": top_warehouses.to_dict('records')
    }
    
    return json.dumps(result, indent=2)

@tool
def get_delays_by_reason(period: str = "all") -> str:
    """
    Get breakdown of delays by reason (Weather, Traffic, Mechanical, Customs, etc.).
    
    Args:
        period: Time period to analyze. Options: 'last week', 'this week', 'last month', 
                'this month', or 'all' for all data
    
    Returns:
        JSON string with delay reasons breakdown
    """
    if period != "all":
        df = data_processor.filter_by_period(period)
    else:
        df = data_processor.get_data()
    
    delayed = df[df['is_delayed']]
    
    if len(delayed) == 0:
        return json.dumps({"message": "No delays found for the specified period"})
    
    delay_summary = delayed.groupby('delay_reason').agg({
        'id': 'count',
        'delay_minutes': ['mean', 'sum']
    }).reset_index()
    
    delay_summary.columns = ['delay_reason', 'count', 'avg_delay_minutes', 'total_delay_minutes']
    delay_summary = delay_summary.sort_values('count', ascending=False)
    
    result = {
        "period": period,
        "total_delayed_shipments": int(len(delayed)),
        "delay_breakdown": delay_summary.to_dict('records')
    }
    
    return json.dumps(result, indent=2)

@tool
def get_shipments_by_filters(warehouse: Optional[str] = None, 
                             route: Optional[str] = None,
                             delay_reason: Optional[str] = None,
                             min_delay: Optional[int] = None,
                             period: str = "all") -> str:
    """
    Filter and retrieve shipments based on multiple criteria.
    
    Args:
        warehouse: Filter by warehouse name (e.g., 'WH1', 'WH2', 'WH3')
        route: Filter by route name (e.g., 'Route A', 'Route B', 'Route C')
        delay_reason: Filter by delay reason (e.g., 'Weather', 'Traffic', 'Mechanical', 'Customs')
        min_delay: Minimum delay in minutes
        period: Time period ('last week', 'this week', 'last month', 'this month', 'all')
    
    Returns:
        JSON string with filtered shipments
    """
    if period != "all":
        df = data_processor.filter_by_period(period)
    else:
        df = data_processor.get_data()
    
    # Apply filters
    if warehouse:
        df = df[df['warehouse'] == warehouse]
    if route:
        df = df[df['route'] == route]
    if delay_reason:
        df = df[df['delay_reason'] == delay_reason]
    if min_delay is not None:
        df = df[df['delay_minutes'] >= min_delay]
    
    if len(df) == 0:
        return json.dumps({"message": "No shipments found matching the criteria"})
    
    # Convert to records and format dates
    shipments = df.to_dict('records')
    for shipment in shipments:
        shipment['date'] = shipment['date'].strftime('%Y-%m-%d')
    
    result = {
        "filters_applied": {
            "warehouse": warehouse,
            "route": route,
            "delay_reason": delay_reason,
            "min_delay_minutes": min_delay,
            "period": period
        },
        "total_shipments_found": len(shipments),
        "shipments": shipments[:20]  # Limit to first 20 for readability
    }
    
    if len(shipments) > 20:
        result["note"] = f"Showing first 20 of {len(shipments)} shipments"
    
    return json.dumps(result, indent=2)

@tool
def get_delivery_time_analysis(warehouse: Optional[str] = None, 
                               route: Optional[str] = None) -> str:
    """
    Analyze average delivery times, optionally filtered by warehouse or route.
    
    Args:
        warehouse: Filter by specific warehouse (optional)
        route: Filter by specific route (optional)
    
    Returns:
        JSON string with delivery time analysis
    """
    df = data_processor.get_data()
    
    # Apply filters
    if warehouse:
        df = df[df['warehouse'] == warehouse]
    if route:
        df = df[df['route'] == route]
    
    if len(df) == 0:
        return json.dumps({"error": "No data found for the specified filters"})
    
    result = {
        "filters": {"warehouse": warehouse, "route": route},
        "average_delivery_time_days": round(df['delivery_time_days'].mean(), 2),
        "min_delivery_time_days": round(df['delivery_time_days'].min(), 2),
        "max_delivery_time_days": round(df['delivery_time_days'].max(), 2),
        "average_processing_time_hours": round(df['processing_time_hours'].mean(), 2),
        "total_shipments_analyzed": len(df)
    }
    
    return json.dumps(result, indent=2)

@tool
def get_on_time_delivery_rate(period: str = "all", 
                              warehouse: Optional[str] = None,
                              route: Optional[str] = None) -> str:
    """
    Calculate on-time delivery rate (percentage of shipments without delays).
    
    Args:
        period: Time period to analyze ('last week', 'this week', 'last month', 'this month', 'all')
        warehouse: Filter by specific warehouse (optional)
        route: Filter by specific route (optional)
    
    Returns:
        JSON string with on-time delivery rate
    """
    if period != "all":
        df = data_processor.filter_by_period(period)
    else:
        df = data_processor.get_data()
    
    # Apply filters
    if warehouse:
        df = df[df['warehouse'] == warehouse]
    if route:
        df = df[df['route'] == route]
    
    if len(df) == 0:
        return json.dumps({"error": "No data found for the specified criteria"})
    
    total_shipments = len(df)
    on_time_shipments = len(df[~df['is_delayed']])
    delayed_shipments = total_shipments - on_time_shipments
    on_time_rate = (on_time_shipments / total_shipments) * 100
    
    result = {
        "period": period,
        "filters": {"warehouse": warehouse, "route": route},
        "total_shipments": total_shipments,
        "on_time_shipments": on_time_shipments,
        "delayed_shipments": delayed_shipments,
        "on_time_rate_percent": round(on_time_rate, 2)
    }
    
    return json.dumps(result, indent=2)

@tool
def get_delay_trend_analysis(period: str = "last week") -> str:
    """
    Analyze delay trends over time to see if delays are increasing or decreasing.
    
    Args:
        period: Time period to analyze ('last week', 'this week', 'last month', 'this month')
    
    Returns:
        JSON string with trend analysis
    """
    df = data_processor.filter_by_period(period)
    
    if len(df) == 0:
        return json.dumps({"error": "No data found for the specified period"})
    
    # Get daily delay averages
    daily_delays = df.groupby('date')['delay_minutes'].mean().reset_index()
    daily_delays = daily_delays.sort_values('date')
    
    # Calculate trend
    if len(daily_delays) > 1:
        X = np.arange(len(daily_delays)).reshape(-1, 1)
        y = daily_delays['delay_minutes'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        slope = model.coef_[0]
        
        if slope > 1:
            trend = "increasing"
        elif slope < -1:
            trend = "decreasing"
        else:
            trend = "stable"
        
        trend_description = f"Average delay is {trend} by approximately {abs(slope):.2f} minutes per day"
    else:
        trend = "insufficient data"
        trend_description = "Not enough data points to determine trend"
    
    result = {
        "period": period,
        "trend": trend,
        "trend_description": trend_description,
        "first_day_avg_delay": round(float(daily_delays.iloc[0]['delay_minutes']), 2),
        "last_day_avg_delay": round(float(daily_delays.iloc[-1]['delay_minutes']), 2),
        "period_avg_delay": round(float(daily_delays['delay_minutes'].mean()), 2),
        "daily_data": daily_delays.to_dict('records')
    }
    
    return json.dumps(result, indent=2, default=str)

@tool
def predict_next_week_delay() -> str:
    """
    Predict the average delay for next week using historical data and linear regression.
    This tool uses historical delay patterns to forecast future delays.
    
    Returns:
        JSON string with prediction results
    """
    try:
        X, y = data_processor.prepare_prediction_data(target='delay_minutes')
        
        if len(X) < 5:
            return json.dumps({"error": "Insufficient historical data for prediction"})
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next 7 days
        last_day = X[-1][0]
        future_days = np.array([[last_day + i] for i in range(1, 8)])
        predictions = model.predict(future_days)
        
        avg_prediction = predictions.mean()
        
        # Calculate confidence (R² score)
        r2_score = model.score(X, y)
        
        result = {
            "prediction_type": "Linear Regression",
            "predicted_avg_delay_next_week_minutes": round(float(avg_prediction), 2),
            "daily_predictions": [round(float(p), 2) for p in predictions],
            "confidence_score": round(float(r2_score), 3),
            "historical_avg_delay_minutes": round(float(y.mean()), 2),
            "note": "Prediction based on historical trend analysis"
        }
        
        return json.dumps(result, indent=2)
    
    except Exception as e:
        return json.dumps({"error": f"Prediction failed: {str(e)}"})

@tool
def get_warehouse_optimization_recommendation() -> str:
    """
    Provide recommendations for warehouse optimization based on performance metrics.
    Analyzes processing times, delay rates, and efficiency to suggest improvements.
    
    Returns:
        JSON string with optimization recommendations
    """
    warehouse_perf = data_processor.get_warehouse_performance()
    
    recommendations = []
    
    for _, wh in warehouse_perf.iterrows():
        issues = []
        suggestions = []
        
        # Check processing time
        if wh['avg_processing_time_hours'] > 25:
            issues.append(f"High processing time ({wh['avg_processing_time_hours']:.1f} hours)")
            suggestions.append("Consider process optimization or additional staff")
        
        # Check delay rate
        if wh['on_time_rate_percent'] < 60:
            issues.append(f"Low on-time rate ({wh['on_time_rate_percent']:.1f}%)")
            suggestions.append("Investigate root causes of delays and implement corrective actions")
        
        # Check average delay
        if wh['avg_delay_minutes'] > 40:
            issues.append(f"High average delay ({wh['avg_delay_minutes']:.1f} minutes)")
            suggestions.append("Review scheduling and resource allocation")
        
        priority = "High" if len(issues) >= 2 else "Medium" if len(issues) == 1 else "Low"
        
        recommendations.append({
            "warehouse": wh['warehouse'],
            "priority": priority,
            "issues": issues if issues else ["No major issues identified"],
            "suggestions": suggestions if suggestions else ["Maintain current performance"],
            "current_metrics": {
                "processing_time_hours": round(wh['avg_processing_time_hours'], 2),
                "on_time_rate_percent": round(wh['on_time_rate_percent'], 2),
                "avg_delay_minutes": round(wh['avg_delay_minutes'], 2)
            }
        })
    
    result = {
        "analysis_date": datetime.now().strftime('%Y-%m-%d'),
        "recommendations": sorted(recommendations, key=lambda x: (x['priority'] == 'Low', x['priority'] == 'Medium', x['priority'] == 'High'), reverse=True)
    }
    
    return json.dumps(result, indent=2)

# List of all tools for the agent
ALL_TOOLS = [
    get_overall_statistics,
    get_most_delayed_route,
    get_top_warehouses_by_processing_time,
    get_delays_by_reason,
    get_shipments_by_filters,
    get_delivery_time_analysis,
    get_on_time_delivery_rate,
    get_delay_trend_analysis,
    predict_next_week_delay,
    get_warehouse_optimization_recommendation
]
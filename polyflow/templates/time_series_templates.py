from typing import Any, Optional, Union
import pandas as pd


def time_series_anomaly_formatter(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    description: str,
    window_size: Optional[int] = None
) -> list[dict[str, str]]:
    """Format prompt for anomaly detection"""
    
    template = [
        {"role": "system", "content": """You are an expert time series analyst. 
        Analyze the provided time series data to detect anomalies based on the given description.
        Return anomaly scores between 0 and 1 for each timestamp."""},
        
        {"role": "user", "content": f"""
        Time series data:
        {df[[time_col, value_col]].to_string()}
        
        Description of anomalies to detect:
        {description}
        
        Window size: {window_size if window_size else 'Full series'}
        
        Analyze this time series and return anomaly scores.
        """}
    ]
    
    return template
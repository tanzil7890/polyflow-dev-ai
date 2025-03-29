from typing import Any, Optional, Union, List, Dict
import pandas as pd
import numpy as np
import polyflow
from polyflow.templates import task_instructions
from polyflow.types import TimeSeriesOutput
from polyflow.models.LM_time import LMTime
import json

# Check if the accessor is already registered to avoid duplicate registration
if not hasattr(pd.DataFrame, 'vecsem_time_series'):
    @pd.api.extensions.register_dataframe_accessor("vecsem_time_series")
    class SemTimeSeriesDataframe:
        """DataFrame accessor for semantic time series analysis."""

        def __init__(self, pandas_obj: Any):
            self._validate(pandas_obj)
            self._obj = pandas_obj

        @staticmethod
        def _validate(obj: Any) -> None:
            if not isinstance(obj, pd.DataFrame):
                raise AttributeError("Must be a DataFrame")

        def detect_anomalies(
            self, 
            time_col: str,
            value_col: str,
            description: str,
            threshold: float = 0.5,
            safe_mode: bool = False
        ) -> pd.DataFrame:
            """Detect anomalies in time series data."""
            if polyflow.settings.lm is None:
                raise ValueError("Language model not configured. Use polyflow.settings.configure()")

            # Calculate statistical measures for anomaly detection
            mean = self._obj[value_col].mean()
            std = self._obj[value_col].std()
            z_scores = (self._obj[value_col] - mean) / std
            
            # Format data for the prompt
            data = pd.DataFrame({
                'timestamp': self._obj[time_col],
                'value': self._obj[value_col],
                'z_score': z_scores
            })
            data_str = data.to_string()
            
            messages = [
                {
                    "role": "system",
                    "content": """You are a financial analyst. For each data point, return a JSON array with objects containing:
                    1. 'score': anomaly score between 0 and 1 (higher means more anomalous)
                    2. 'explanation': brief explanation of why this point is or isn't anomalous
                    
                    Use these rules:
                    - z-score > 2 or < -2: score > 0.7
                    - Sudden changes (>5% in one day): score > 0.6
                    - Breaking moving averages: score > 0.5
                    - Normal movements: score < 0.3"""
                },
                {
                    "role": "user",
                    "content": f"""Analyze this stock price data for anomalies:
                    {data_str}
                    
                    Context: {description}
                    
                    Return JSON array with one object per data point."""
                }
            ]

            try:
                # Make LM call
                response = polyflow.settings.lm(messages, safe_mode=safe_mode)
                
                # Process response
                if isinstance(response, str):
                    try:
                        results = json.loads(response)
                    except json.JSONDecodeError:
                        # Try to clean and parse the response
                        cleaned_response = response.strip()
                        if cleaned_response.startswith("[") and cleaned_response.endswith("]"):
                            try:
                                results = json.loads(cleaned_response)
                            except:
                                results = None
                        else:
                            results = None
                else:
                    results = response
                    
                if not results:
                    print("No valid anomalies detected")
                    return pd.DataFrame()
                    
                # Create results DataFrame
                df_results = pd.DataFrame({
                    'timestamp': self._obj[time_col],
                    'value': self._obj[value_col],
                    'anomaly_score': [float(r.get('score', 0)) for r in results],
                    'anomaly_explanation': [str(r.get('explanation', '')) for r in results],
                    'is_anomaly': [float(r.get('score', 0)) >= threshold for r in results]
                })
                
                return df_results
                
            except Exception as e:
                if safe_mode:
                    print(f"Warning: {str(e)}")
                    return pd.DataFrame()
                raise e

        def forecast(
            self,
            time_col: str,
            value_col: str, 
            horizon: int,
            context: str,
            confidence_intervals: bool = True,
            interval_width: float = 0.95,
            safe_mode: bool = False
        ) -> pd.DataFrame:
            """
            Generate semantic time series forecasts.

            Args:
                time_col: Column containing timestamps
                value_col: Column containing values to forecast
                horizon: Number of periods to forecast
                context: Natural language context for forecasting
                confidence_intervals: Whether to include prediction intervals
                interval_width: Width of confidence intervals (0 to 1)
                safe_mode: Whether to show cost estimate before running

            Returns:
                DataFrame with forecasts and optional confidence intervals
            """
            if polyflow.settings.lm is None:
                raise ValueError(
                    "Language model not configured. Use polyflow.settings.configure()"
                )

            # Validate inputs
            if time_col not in self._obj.columns:
                raise ValueError(f"Time column '{time_col}' not found in DataFrame")
            if value_col not in self._obj.columns:
                raise ValueError(f"Value column '{value_col}' not found in DataFrame")

            # Format prompt for forecasting
            prompt = [
                {"role": "system", "content": """You are an expert time series forecaster.
                Generate forecasts based on historical data and provided context.
                Consider trends, seasonality, and relevant patterns."""},
                
                {"role": "user", "content": f"""
                Historical data:
                {self._obj[[time_col, value_col]].to_string()}
                
                Forecast horizon: {horizon} periods
                Context: {context}
                Include confidence intervals: {confidence_intervals}
                Confidence level: {interval_width}
                
                For each future period, provide:
                1. Forecast value
                2. Lower bound (if confidence intervals requested)
                3. Upper bound (if confidence intervals requested)
                4. Brief explanation
                
                Format: JSON array of forecast objects
                """}
            ]

            # Get LLM response
            response = polyflow.settings.lm(
                prompt, 
                safe_mode=safe_mode,
                progress_bar_desc="Generating forecast"
            )

            # Process results
            forecasts = self._process_forecasts(response, confidence_intervals)
            
            return forecasts

        def find_patterns(
            self,
            time_col: str,
            value_col: str,
            pattern_desc: str,
            min_pattern_length: int = 2,
            safe_mode: bool = False
        ) -> pd.DataFrame:
            """
            Find patterns in time series matching semantic description.

            Args:
                time_col: Column containing timestamps
                value_col: Column containing values to analyze
                pattern_desc: Natural language description of patterns to find
                min_pattern_length: Minimum length of patterns to detect
                safe_mode: Whether to show cost estimate before running

            Returns:
                DataFrame with pattern matches and scores
            """
            if polyflow.settings.lm is None:
                raise ValueError(
                    "Language model not configured. Use polyflow.settings.configure()"
                )

            # Validate inputs
            if time_col not in self._obj.columns:
                raise ValueError(f"Time column '{time_col}' not found in DataFrame")
            if value_col not in self._obj.columns:
                raise ValueError(f"Value column '{value_col}' not found in DataFrame")

            # Format prompt for pattern matching
            prompt = [
                {"role": "system", "content": """You are an expert in time series pattern recognition.
                Identify patterns in the data that match the given description.
                Consider temporal relationships and value characteristics."""},
                
                {"role": "user", "content": f"""
                Time series data:
                {self._obj[[time_col, value_col]].to_string()}
                
                Pattern description: {pattern_desc}
                Minimum pattern length: {min_pattern_length}
                
                For each pattern found, provide:
                1. Start timestamp
                2. End timestamp
                3. Pattern score (0-1)
                4. Pattern description
                
                Format: JSON array of pattern objects
                """}
            ]

            # Get LLM response
            response = polyflow.settings.lm(
                prompt,
                safe_mode=safe_mode, 
                progress_bar_desc="Finding patterns"
            )

            # Process results
            patterns = self._process_patterns(response)
            
            return patterns

        def _process_anomaly_detection(self, response: Any, expected_length: int) -> List[Dict[str, Union[float, str]]]:
            """Process LLM response for anomaly detection"""
            try:
                # Handle empty or None response
                if not response:
                    return [{'score': 0.0, 'explanation': 'No anomaly detected'} for _ in range(expected_length)]
                
                # Ensure response is a list
                results = response if isinstance(response, list) else [response]
                
                # Validate and clean results
                processed_results = []
                for result in results:
                    if isinstance(result, dict):
                        processed_results.append({
                            'score': float(result.get('score', 0)),
                            'explanation': str(result.get('explanation', 'No explanation provided'))
                        })
                
                # Ensure we have the correct number of results
                if len(processed_results) < expected_length:
                    # Pad with default values
                    processed_results.extend([
                        {'score': 0.0, 'explanation': 'No anomaly detected'} 
                        for _ in range(expected_length - len(processed_results))
                    ])
                elif len(processed_results) > expected_length:
                    # Truncate to expected length
                    processed_results = processed_results[:expected_length]
                    
                return processed_results
                
            except Exception as e:
                if hasattr(polyflow.settings.lm, 'safe_mode') and polyflow.settings.lm.safe_mode:
                    print(f"Warning: Failed to process anomaly detection results: {str(e)}")
                    return [{'score': 0.0, 'explanation': 'Processing error'} for _ in range(expected_length)]
                raise ValueError(f"Failed to process anomaly detection results: {str(e)}")

        def _process_forecasts(
            self, 
            response: Any,
            confidence_intervals: bool
        ) -> pd.DataFrame:
            """Process LLM response for forecasting"""
            try:
                # Response should already be parsed JSON from LMTime.__call__
                forecasts = response if isinstance(response, list) else [response]
                
                # Create forecast DataFrame
                forecast_data = {
                    'forecast': [],
                    'explanation': []
                }
                
                if confidence_intervals:
                    forecast_data['lower_bound'] = []
                    forecast_data['upper_bound'] = []
                
                for f in forecasts:
                    forecast_data['forecast'].append(float(f.get('value', 0)))
                    forecast_data['explanation'].append(str(f.get('explanation', '')))
                    
                    if confidence_intervals:
                        forecast_data['lower_bound'].append(float(f.get('lower_bound', 0)))
                        forecast_data['upper_bound'].append(float(f.get('upper_bound', 0)))
                
                return pd.DataFrame(forecast_data)
                
            except Exception as e:
                raise ValueError(f"Failed to process forecast results: {str(e)}")

        def _process_patterns(self, response: Any) -> pd.DataFrame:
            """Process LLM response for pattern matching"""
            try:
                # Parse JSON response
                patterns = response.json()
                
                # Create patterns DataFrame
                pattern_data = {
                    'start_time': [],
                    'end_time': [],
                    'pattern_score': [],
                    'description': []
                }
                
                for p in patterns:
                    pattern_data['start_time'].append(p['start_timestamp'])
                    pattern_data['end_time'].append(p['end_timestamp'])
                    pattern_data['pattern_score'].append(float(p['score']))
                    pattern_data['description'].append(str(p['description']))
                
                return pd.DataFrame(pattern_data)
                
            except Exception as e:
                raise ValueError(f"Failed to process pattern matching results: {str(e)}")

# If running this file directly, don't do anything
if __name__ == '__main__':
    print("This module is not meant to be run directly.")
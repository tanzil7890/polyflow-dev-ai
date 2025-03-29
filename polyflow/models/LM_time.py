from polyflow.models.language_processor import LanguageProcessor

class LMTime(LanguageProcessor):
    """
    A language model for time series analysis.
    This extends the base LanguageProcessor with time series specific functionality.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def analyze_time_series(self, timestamps, values, description=None):
        """
        Analyze a time series for patterns, anomalies, or forecasts.
        
        Args:
            timestamps: List of timestamps
            values: List of corresponding values
            description: Optional description of the time series
            
        Returns:
            Analysis results as a dictionary
        """
        # Format the data for analysis
        data_points = "\n".join([f"{t}: {v}" for t, v in zip(timestamps, values)])
        
        # Create the prompt for time series analysis
        messages = [
            {
                "role": "system",
                "content": "You are a time series analysis expert. Analyze the provided time series data."
            },
            {
                "role": "user",
                "content": f"Here is the time series data:\n{data_points}\n\n"
                          f"Description: {description or 'No description provided.'}\n\n"
                          "Please analyze this data for patterns, trends, seasonality, and anomalies."
            }
        ]
        
        # Get the response
        response = self.generate(messages)
        return response 
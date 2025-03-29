import litellm
from typing import Any, Optional, Dict, List, Union
import json
from polyflow.types import LMOutput

class TemporalLanguageProcessor:
    """
    Specialized language model processor for time series analysis tasks.
    Handles anomaly detection, forecasting, and pattern identification in time series data.
    """
    
    def __init__(
        self, 
        model_identifier: str, 
        api_credentials: str, 
        response_max_length: int = 1024,
        generation_temperature: float = 0.1,
        **model_params
    ):
        """
        Initialize the temporal language processor.
        
        Args:
            model_identifier: Model to use for time series analysis
            api_credentials: API key for model access
            response_max_length: Maximum output token length
            generation_temperature: Controls randomness in generation
            model_params: Additional parameters for the model
        """
        self.model_name = model_identifier
        self.credentials = api_credentials
        self.configuration = {
            "max_tokens": response_max_length,
            "temperature": generation_temperature,
            **model_params
        }
        self.token_count = 0
        self.usage_cost = 0.0

    def generate(
        self, 
        message_lists: List[List[Dict[str, str]]], 
        error_recovery: bool = False,
        progress_indicator: Optional[str] = None
    ) -> LMOutput:
        """
        Process time series analysis prompts and return structured results.
        
        Args:
            message_lists: Lists of message dictionaries
            error_recovery: If True, return empty results on error
            progress_indicator: Optional description for progress tracking
            
        Returns:
            LMOutput containing analysis results
        """
        all_outputs = []
        
        for messages in message_lists:
            try:
                # Validate input format
                if not isinstance(messages, list):
                    raise ValueError("Messages must be a list of dictionaries")

                # Configure request parameters
                request_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "api_key": self.credentials,
                    **self.configuration
                }

                # Execute model call
                response = litellm.completion(**request_params)

                # Track usage statistics
                if hasattr(response, 'usage'):
                    self.token_count += response.usage.total_tokens

                # Extract and process content
                if hasattr(response, 'choices') and response.choices:
                    raw_content = response.choices[0].message.content.strip()
                    try:
                        # Clean content for better JSON parsing
                        cleaned_content = self._clean_json_content(raw_content)
                        all_outputs.append(json.loads(cleaned_content))
                    except json.JSONDecodeError as e:
                        if error_recovery:
                            print(f"Warning: JSON parsing failed: {str(e)}")
                            print(f"Content preview: {raw_content[:100]}...")
                            all_outputs.append([])
                        else:
                            raise ValueError(f"Invalid JSON in model response: {str(e)}")
                else:
                    all_outputs.append(None)

            except Exception as e:
                if error_recovery:
                    print(f"Error in time series analysis: {str(e)}")
                    all_outputs.append(None)
                else:
                    raise e
                    
        return LMOutput(outputs=all_outputs)
    
    def _clean_json_content(self, content: str) -> str:
        """
        Clean and extract JSON content from model output.
        
        Args:
            content: Raw content from the model
            
        Returns:
            Cleaned JSON string ready for parsing
        """
        # Basic cleaning
        cleaned = content.replace('\n', '').replace('    ', '')
        
        # Extract JSON array if embedded in other text
        if not cleaned.startswith('['):
            # Find the JSON array boundaries
            start_idx = cleaned.find('[')
            end_idx = cleaned.rfind(']')
            if start_idx != -1 and end_idx != -1:
                cleaned = cleaned[start_idx:end_idx + 1]
                
        return cleaned

    def display_usage_summary(self) -> None:
        """Display token usage and cost statistics."""
        print(f"\nTotal tokens used: {self.token_count}")
        if self.usage_cost > 0:
            print(f"Total cost: ${self.usage_cost:.4f}")

    def create_time_series_prompt(
        self,
        time_series_data: str,
        analysis_task: str,
        task_description: str,
        task_parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        Create a specialized prompt for time series analysis.
        
        Args:
            time_series_data: String representation of time series data
            analysis_task: Type of analysis ("anomaly", "forecast", or "pattern")
            task_description: Specific requirements for the task
            task_parameters: Additional parameters for analysis
            
        Returns:
            Formatted message list ready for the model
        """
        # Task-specific descriptions
        task_templates = {
            "anomaly": "Analyze the provided time series data to detect anomalies.",
            "forecast": "Generate forecasts based on the provided time series data.",
            "pattern": "Identify patterns in the provided time series data."
        }

        # Create system and user messages
        system_message = f"You are an expert time series analyst specialized in {analysis_task} detection."
        
        user_message = f"""Please analyze this time series data:

{time_series_data}

Task: {task_templates.get(analysis_task, "Analyze")}
Description: {task_description}

"""

        # Add any additional parameters
        if task_parameters:
            user_message += "\nAdditional parameters:\n"
            user_message += "\n".join(f"- {k}: {v}" for k, v in task_parameters.items())

        # Return formatted messages
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

    def parse_structured_response(self, response: Any) -> Union[List[Dict], Dict]:
        """
        Parse structured data from model response.
        
        Args:
            response: Raw model response
            
        Returns:
            Parsed JSON data
        """
        try:
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                return json.loads(content)
            return {}
        except Exception as e:
            raise ValueError(f"Failed to parse model response: {str(e)}")
            
    # Alias methods for compatibility
    def __call__(self, messages, safe_mode=False, progress_bar_desc=None):
        return self.generate([messages], safe_mode, progress_bar_desc)
        
    def print_total_usage(self):
        return self.display_usage_summary()
        
    def _format_time_series_prompt(self, data_str, task_type, description, additional_params=None):
        return self.create_time_series_prompt(data_str, task_type, description, additional_params)
        
    def _parse_json_response(self, response):
        return self.parse_structured_response(response)
import pandas as pd
import json
from typing import List, Optional, Dict, Any
from polyflow.models import LM
from polyflow.settings import settings

@pd.api.extensions.register_dataframe_accessor("sem_classify")
class SemClassifyAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._settings = settings
        
    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse LM response into a standardized format"""
        if isinstance(response, dict):
            return response
        if isinstance(response, str):
            # Remove any markdown code block indicators
            response = response.replace('```json', '').replace('```', '').strip()
            try:
                # Try to parse as JSON
                parsed = json.loads(response)
                # If the response is already in our format, return it
                if all(k in parsed for k in ['category', 'confidence', 'reasoning']):
                    return parsed
                # If it's a different JSON format, try to extract relevant fields
                return {
                    'category': parsed.get('category', 'Unknown'),
                    'confidence': float(parsed.get('confidence', 0.0)),
                    'reasoning': parsed.get('reasoning', response)
                }
            except json.JSONDecodeError:
                # If not JSON, try to extract category from text
                text = response.lower()
                for category in self._categories:
                    if category.lower() in text:
                        # Extract just the reasoning part if it's embedded in JSON
                        reasoning = text
                        if '"reasoning":' in text:
                            try:
                                reasoning = text.split('"reasoning":')[1].split('"')[1]
                            except:
                                pass
                        return {
                            'category': category,
                            'confidence': 1.0,
                            'reasoning': reasoning
                        }
                # Default fallback
                return {
                    'category': 'Unknown',
                    'confidence': 0.0,
                    'reasoning': response
                }
        return None

    def __call__(
        self,
        text_column: str,
        categories: List[str],
        prompt_template: Optional[str] = None,
        return_scores: bool = False,
        return_reasoning: bool = False
    ) -> pd.DataFrame:
        """
        Classify text data into predefined categories using semantic analysis.
        """
        if not self._settings.lm:
            raise ValueError("LM model not configured. Use polyflow.settings.configure()")
            
        self._categories = categories  # Store categories for response parsing
            
        # Default prompt template if none provided
        if not prompt_template:
            prompt_template = (
                "Classify the following text into one of these categories: "
                f"{', '.join(categories)}.\n\nText: {{text}}\n\n"
                "Provide your response in JSON format with the following fields:\n"
                "- category: the selected category\n"
                "- confidence: confidence score between 0 and 1\n"
                "- reasoning: brief explanation for the classification"
            )
            
        results = []
        for text in self._obj[text_column]:
            messages = [[{
                "role": "system",
                "content": "You are an expert at classifying text into predefined categories. Always respond with valid JSON."
            }, {
                "role": "user",
                "content": prompt_template.format(text=text)
            }]]
            
            try:
                response = self._settings.lm(messages)
                result = self._parse_response(response.outputs[0]) if hasattr(response, 'outputs') else None
                results.append(result)
            except Exception as e:
                print(f"Error processing text: {str(e)}")
                results.append(None)
                
        # Create output DataFrame
        df = self._obj.copy()
        df['classification'] = [r['category'] if r else None for r in results]
        
        if return_scores:
            df['confidence_score'] = [r['confidence'] if r else None for r in results]
            
        if return_reasoning:
            df['classification_reasoning'] = [r['reasoning'] if r else None for r in results]
            
        return df

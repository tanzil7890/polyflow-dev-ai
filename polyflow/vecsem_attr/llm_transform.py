from typing import Any, Optional
import pandas as pd
import numpy as np
import json
import re

import polyflow
from polyflow.models.lm import LM

@pd.api.extensions.register_dataframe_accessor("llm_transform")
class SemTransformDataframe:
    """DataFrame accessor for LLM-based semantic transformations."""

    def __init__(self, pandas_obj: Any) -> None:
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def __call__(
        self,
        column: str,
        transform_query: str,
        threshold: float = 0.7,
        suffix: str = "_transformed",
        return_scores: bool = False,
        return_reasoning: bool = True,
        system_prompt: Optional[str] = None
    ) -> pd.DataFrame:
        """Transform column values using LLM-based semantic understanding."""
        if polyflow.settings.lm is None:
            raise ValueError(
                "The language model must be configured using polyflow.settings.configure()"
            )

        lm = polyflow.settings.lm
        if not isinstance(lm, LM):
            raise ValueError("Invalid language model type")

        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = (
                "You are an expert at analyzing and classifying data. For each input, "
                "determine if it matches the given criteria and respond with a JSON object containing:\n"
                "1. 'score': A confidence score between 0 and 1\n"
                "2. 'transformed': Either:\n"
                "   - The original value or 'Yes' if it matches (score >= threshold)\n"
                "   - null if it doesn't match (score < threshold)\n"
                "3. 'reasoning': A brief explanation of your decision\n\n"
                "Example response for a leadership role:\n"
                "{\n"
                "  'score': 0.92,\n"
                "  'transformed': 'Yes',\n"
                "  'reasoning': 'This is a senior management position with leadership responsibilities'\n"
                "}\n\n"
                "Be strict with scoring and only assign high scores (>0.8) when there's strong evidence."
            )

        # Process each row
        results = []
        scores = []
        reasonings = []
        
        for value in self._obj[column]:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": (
                    f"Input: {value}\n"
                    f"Criteria: {transform_query}\n"
                    f"Threshold: {threshold}\n"
                    "Provide your analysis as a JSON object with score, transformed value, and reasoning."
                )}
            ]
            
            try:
                response = lm([messages])
                result_text = response.outputs[0]
                
                try:
                    json_match = re.search(r'\{[^{}]*\}', result_text)
                    if json_match:
                        json_str = json_match.group()
                        result = json.loads(json_str)
                        
                        score = float(result.get("score", 0))
                        score = max(0.0, min(1.0, score))
                        
                        transformed = result.get("transformed")
                        reasoning = result.get("reasoning", "No reasoning provided")
                        
                        if score >= threshold:
                            transformed = transformed if transformed not in ["Yes", "yes", "TRUE", "true"] else "Yes"
                        else:
                            transformed = None
                            
                        results.append(transformed)
                        scores.append(score if score >= threshold else None)
                        reasonings.append(reasoning)
                    else:
                        results.append(None)
                        scores.append(None)
                        reasonings.append("Failed to parse response")
                except Exception as e:
                    results.append(None)
                    scores.append(None)
                    reasonings.append(f"Error: {str(e)}")
            except Exception as e:
                results.append(None)
                scores.append(None)
                reasonings.append(f"Error: {str(e)}")
        
        # Create transformed DataFrame
        transformed = self._obj.copy()
        transformed[f"{column}{suffix}"] = pd.Series(
            results,
            index=self._obj.index,
            dtype="string[python]"
        )
        
        if return_scores:
            transformed[f"{column}_score"] = pd.Series(
                scores,
                index=self._obj.index,
                dtype="float64"
            ).round(2)
            
        if return_reasoning:
            transformed[f"{column}_reasoning"] = pd.Series(
                reasonings,
                index=self._obj.index,
                dtype="string[python]"
            )
            
        return transformed

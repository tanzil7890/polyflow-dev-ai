import pandas as pd
import json
import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from polyflow.models import LM
from polyflow.settings import settings

@pd.api.extensions.register_dataframe_accessor("sem_hybrid_classify")
class SemHybridClassifyAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._settings = settings
        
    def _parse_response(self, response: Any, categories: List[str]) -> Dict[str, Any]:
        """Enhanced response parsing with category validation"""
        try:
            if isinstance(response, dict):
                return response
            if isinstance(response, str):
                # Clean up the response string
                response = response.replace('```json', '').replace('```', '').strip()
                
                # Handle potential nested JSON strings
                if response.startswith('"') and response.endswith('"'):
                    response = response[1:-1].replace('\\', '')
                
                parsed = json.loads(response)
                
                # Validate and normalize the response
                category = str(parsed.get('category', categories[0])).strip()
                confidence = float(parsed.get('confidence', 0.5))
                reasoning = str(parsed.get('reasoning', 'No reasoning provided')).strip()
                
                # Validate category
                if category not in categories:
                    print(f"Warning: Invalid category '{category}'. Defaulting to {categories[0]}")
                    category = categories[0]
                    confidence *= 0.5
                
                return {
                    'category': category,
                    'confidence': min(max(confidence, 0.0), 1.0),  # Ensure confidence is between 0 and 1
                    'reasoning': reasoning
                }
        except Exception as e:
            print(f"Response parsing error: {str(e)}")
            return {
                'category': categories[0],
                'confidence': 0.0,
                'reasoning': f"Failed to parse response: {str(e)}"
            }

    def __call__(
        self,
        text_column: str,
        categories: List[str],
        prompt_template: Optional[str] = None,
        use_embeddings: bool = True,
        threshold: float = 0.5,
        return_scores: bool = False,
        return_reasoning: bool = False
    ) -> pd.DataFrame:
        """
        Advanced hybrid classification using both LM and embedding models.
        """
        if not self._settings.lm or (use_embeddings and not self._settings.rm):
            raise ValueError("Models not configured. Use polyflow.settings.configure()")
            
        # Get embeddings for categories if using RM
        if use_embeddings:
            category_embeddings = self._settings.rm.transformer.encode(
                categories,
                normalize_embeddings=self._settings.rm.normalize_embeddings,
                show_progress_bar=False
            )
        else:
            category_embeddings = None
            
        # Default prompt template with structured output requirement
        if not prompt_template:
            prompt_template = (
                "Classify the following text into one of these categories: "
                f"{', '.join(categories)}.\n\nText: {{text}}\n\n"
                "Respond ONLY with a JSON object containing:\n"
                "- category: exactly one of the provided categories\n"
                "- confidence: number between 0 and 1\n"
                "- reasoning: brief explanation\n\n"
                "Example response:\n"
                '{"category": "Bug", "confidence": 0.95, "reasoning": "Clear description of a technical issue"}'
            )
            
        results = []
        for text in self._obj[text_column]:
            result = self._classify_text(text, categories, category_embeddings, prompt_template, threshold)
            results.append(result)
                
        # Create output DataFrame with enhanced fields
        df = self._obj.copy()
        df['classification'] = [r.get('category') if r else None for r in results]
        
        if return_scores:
            df['confidence_score'] = [r.get('confidence') if r else None for r in results]
            if use_embeddings:
                df['embedding_similarity'] = [r.get('embedding_similarity') if r else None for r in results]
                # Add model agreement score when both models are used
                df['model_agreement'] = [
                    1.0 if r.get('category') == r.get('embedding_category') else 0.5 
                    for r in results
                ]
            
        if return_reasoning:
            df['classification_reasoning'] = [r.get('reasoning') if r else None for r in results]
            
        return df

    def _classify_text(
        self,
        text: str,
        categories: List[str],
        category_embeddings: Optional[np.ndarray],
        prompt_template: str,
        threshold: float
    ) -> Dict[str, Any]:
        """Classify single text using both models with sophisticated scoring"""
        try:
            # Get LM classification
            messages = [[{
                "role": "system",
                "content": (
                    "You are an expert at classifying text. "
                    "Respond ONLY with a valid JSON object containing 'category', 'confidence', and 'reasoning' fields. "
                    "No other text or formatting."
                )
            }, {
                "role": "user",
                "content": prompt_template.format(text=text)
            }]]
            
            lm_response = self._settings.lm(messages)
            result = self._parse_response(lm_response.outputs[0], categories)
            
            if not result:
                result = {
                    'category': categories[0],
                    'confidence': 0.0,
                    'reasoning': 'Failed to parse LM response'
                }
            
            # Get embedding similarity if using RM
            if category_embeddings is not None:
                text_embedding = self._settings.rm.transformer.encode(
                    [text],
                    normalize_embeddings=self._settings.rm.normalize_embeddings,
                    show_progress_bar=False
                ).squeeze()
                
                text_embedding = text_embedding.reshape(1, -1)
                similarities = cosine_similarity(text_embedding, category_embeddings)[0]
                best_match_idx = np.argmax(similarities)
                embedding_similarity = similarities[best_match_idx]
                
                # Store embedding results
                result['embedding_similarity'] = float(embedding_similarity)
                result['embedding_category'] = categories[best_match_idx]
                
                # Adjust confidence based on model agreement
                if embedding_similarity > threshold:
                    if categories[best_match_idx] == result['category']:
                        result['confidence'] = max(result['confidence'], embedding_similarity)
                    else:
                        result['confidence'] *= 0.8  # Reduce confidence when models disagree
                
            return result
            
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return {
                'category': categories[0],
                'confidence': 0.0,
                'reasoning': f"Error: {str(e)}",
                'embedding_similarity': 0.0 if category_embeddings is not None else None
            }

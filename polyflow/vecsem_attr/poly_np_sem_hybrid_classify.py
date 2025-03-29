import pandas as pd
import json
import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from polyflow.models import LM
from polyflow.settings import settings
from polyflow.utils import compute_cosine_similarity


"""
This code uses the Polyflow library for semantic hybrid classification.
"""

@pd.api.extensions.register_dataframe_accessor("poly_np_sem_hybrid_classify")
class PolyNpSemHybridClassifyAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._settings = settings
        #self.logger = get_logger(__name__)
        
    def _parse_response(self, response: Any, categories: List[str]) -> Dict[str, Any]:
        """Enhanced response parsing with Polyflow's JSON handling"""
        try:
            if isinstance(response, dict):
                return response
            if isinstance(response, str):
                # Use Polyflow's JSON parser
                response = self.clean_json_response(response)
                parsed = self.parse_json_safely(response)
                
                # Validate using Polyflow's category validator
                category = self.validate_category(parsed.get('category', ''), categories)
                confidence = self.validate_confidence(parsed.get('confidence', 0.5))
                reasoning = self.validate_reasoning(parsed.get('reasoning', ''))
                
                return {
                    'category': category,
                    'confidence': confidence,
                    'reasoning': reasoning
                }
        except Exception as e:
            self.logger.error(f"Response parsing error: {str(e)}")
            return {
                'category': categories[0],
                'confidence': 0.0,
                'reasoning': f"Failed to parse response: {str(e)}"
            }

    def _classify_text(
        self,
        text: str,
        categories: List[str],
        category_embeddings: Optional[np.ndarray],
        prompt_template: str,
        threshold: float
    ) -> Dict[str, Any]:
        """Classify text using Polyflow's models and utilities"""
        try:
            # Use Polyflow's system prompt generator
            system_prompt = self.generate_system_prompt(
                task="classification",
                format="json",
                categories=categories
            )

            # Get LM classification using Polyflow's LM handler
            messages = self.create_messages(
                system_prompt=system_prompt,
                user_prompt=prompt_template.format(text=text)
            )
            
            lm_response = self._settings.lm(messages)
            result = self._parse_response(lm_response.outputs[0], categories)
            
            # Get embedding similarity using Polyflow's embedding handler
            if category_embeddings is not None:
                text_embedding = self._settings.rm.encode(
                    [text],
                    normalize=True,
                    batch_size=1
                )
                
                # Use Polyflow's similarity function
                similarities = cosine_similarity(
                    text_embedding,
                    category_embeddings
                )
                
                best_match_idx = self.get_best_match(similarities)
                embedding_similarity = float(similarities[best_match_idx])
                
                # Enhanced scoring using Polyflow's confidence calculator
                result.update(
                    self.calculate_confidence_scores(
                        lm_category=result['category'],
                        embedding_category=categories[best_match_idx],
                        lm_confidence=result['confidence'],
                        embedding_similarity=embedding_similarity,
                        threshold=threshold
                    )
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing text: {str(e)}")
            return self.create_error_response(categories[0], str(e))

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
        """Advanced hybrid classification using Polyflow's pipeline"""
        # Validate settings
        self.validate_settings(use_embeddings)
            
        # Get embeddings using Polyflow's batch processor
        category_embeddings = (
            self.get_category_embeddings(categories)
            if use_embeddings else None
        )
            
        # Use Polyflow's prompt template if none provided
        if not prompt_template:
            prompt_template = self.get_default_prompt_template(categories)
            
        # Process texts using Polyflow's batch processor
        results = self.process_batch(
            texts=self._obj[text_column],
            categories=categories,
            category_embeddings=category_embeddings,
            prompt_template=prompt_template,
            threshold=threshold
        )
                
        # Create output DataFrame using Polyflow's results formatter
        return self.format_results(
            df=self._obj.copy(),
            results=results,
            return_scores=return_scores,
            return_reasoning=return_reasoning,
            use_embeddings=use_embeddings
        )

    # Polyflow utility methods
    def validate_settings(self, use_embeddings: bool):
        """Validate Polyflow settings"""
        if not self._settings.lm or (use_embeddings and not self._settings.rm):
            raise ValueError("Models not configured. Use polyflow.settings.configure()")

    def get_category_embeddings(self, categories: List[str]) -> np.ndarray:
        """Get embeddings using Polyflow's embedding model"""
        return self._settings.rm.encode(
            categories,
            normalize=True,
            batch_size=len(categories)
        )

    def get_default_prompt_template(self, categories: List[str]) -> str:
        """Get Polyflow's default prompt template"""
        return self.load_prompt_template(
            template_name="classification",
            categories=categories
        )

    def process_batch(self, **kwargs) -> List[Dict[str, Any]]:
        """Process texts using Polyflow's batch processor"""
        return [
            self._classify_text(
                text=text,
                categories=kwargs['categories'],
                category_embeddings=kwargs['category_embeddings'],
                prompt_template=kwargs['prompt_template'],
                threshold=kwargs['threshold']
            )
            for text in kwargs['texts']
        ]

    def format_results(self, **kwargs) -> pd.DataFrame:
        """Format results using Polyflow's formatter"""
        df = kwargs['df']
        results = kwargs['results']
        
        df['classification'] = [r.get('category') for r in results]
        
        if kwargs['return_scores']:
            df['confidence_score'] = [r.get('confidence') for r in results]
            if kwargs['use_embeddings']:
                df['embedding_similarity'] = [r.get('embedding_similarity') for r in results]
                df['model_agreement'] = [r.get('model_agreement') for r in results]
            
        if kwargs['return_reasoning']:
            df['classification_reasoning'] = [r.get('reasoning') for r in results]
            
        return df
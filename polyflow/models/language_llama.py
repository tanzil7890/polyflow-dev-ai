from typing import List, Dict, Any, Optional, Union
from llama_cpp import Llama
import json
import re
from polyflow.types import LMOutput


class LlamaProcessor:
    """
    Language model processor built on llama.cpp for local LLM inference.
    Provides a standardized interface for working with local Llama models.
    """
    
    def __init__(
        self,
        model_path: str,
        generation_temperature: float = 0.1,
        response_max_tokens: int = 2048,
        context_size: int = 4096,
        gpu_layer_count: int = 1  # For hardware acceleration
    ):
        """
        Initialize the Llama processor.
        
        Args:
            model_path: Path to the Llama model file
            generation_temperature: Controls randomness in generation
            response_max_tokens: Maximum tokens to generate
            context_size: Maximum context window size
            gpu_layer_count: Number of layers to offload to GPU
        """
        # Initialize Llama model
        self.llama_engine = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_gpu_layers=gpu_layer_count,
            verbose=False
        )
        
        # Store configuration
        self.generation_temperature = generation_temperature
        self.response_max_tokens = response_max_tokens
        self.token_usage_count = 0
        
    def generate(
        self, 
        message_lists: List[List[Dict[str, str]]], 
        fallback_enabled: bool = False,
        **generation_params: Dict[str, Any]
    ) -> LMOutput:
        """
        Process multiple message chains and return responses.
        
        Args:
            message_lists: List of message chains to process
            fallback_enabled: Whether to use fallback values on error
            generation_params: Additional parameters for generation
            
        Returns:
            LMOutput containing generated responses
        """
        outputs = []
        
        for messages in message_lists:
            try:
                # Format messages for Llama
                formatted_prompt = self.prepare_prompt(messages)
                
                # Generate response
                response = self.llama_engine(
                    formatted_prompt,
                    max_tokens=self.response_max_tokens,
                    temperature=self.generation_temperature,
                    echo=False
                )
                
                # Track token usage
                if 'usage' in response:
                    self.token_usage_count += response['usage']['total_tokens']
                
                # Extract response text
                response_text = response['choices'][0]['text'].strip()
                
                # Try to parse as JSON
                try:
                    structured_response = json.loads(response_text)
                    outputs.append(structured_response)
                except json.JSONDecodeError:
                    if fallback_enabled:
                        # Extract score from text response if it's a scoring task
                        outputs.append(self.extract_numerical_score(response_text))
                    else:
                        outputs.append(response_text)
                    
            except Exception as e:
                if fallback_enabled:
                    print(f"Warning: {str(e)}")
                    outputs.append({"score": 0.5})  # Default fallback score
                else:
                    raise e
                    
        return LMOutput(outputs=outputs)
    
    def prepare_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages for Llama chat format.
        
        Args:
            messages: List of role-based messages
            
        Returns:
            Formatted prompt string for Llama
        """
        formatted_text = ""
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted_text += f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n"
            elif role == "user":
                formatted_text += f"{content} [/INST]\nReturn a JSON response with a 'score' field.\n"
            elif role == "assistant":
                formatted_text += f"{content}</s>"
                
        return formatted_text
    
    def extract_numerical_score(self, text: str) -> Dict[str, float]:
        """
        Extract numerical score from text response.
        
        Args:
            text: Response text to parse
            
        Returns:
            Dictionary with extracted score
        """
        # Look for numbers in the text
        number_patterns = re.findall(r"(?:score|confidence|relationship)?\s*(?:of|:)?\s*(\d*\.?\d+)", text.lower())
        
        if number_patterns:
            raw_score = float(number_patterns[0])
            # Ensure score is between 0 and 1
            normalized_score = max(0.0, min(1.0, raw_score))
            return {"score": normalized_score}
            
        # Default score if no number found
        return {"score": 0.5}
    
    def display_usage_summary(self) -> None:
        """Display token usage statistics."""
        print(f"Total tokens used: {self.token_usage_count}")
        
    # Alias methods for compatibility
    def __call__(self, messages, safe_mode=False, **kwargs):
        return self.generate([messages], fallback_enabled=safe_mode, **kwargs)
        
    def print_total_usage(self):
        self.display_usage_summary()
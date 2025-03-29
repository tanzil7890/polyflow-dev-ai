import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import litellm
import numpy as np
from litellm import batch_completion, completion_cost
from litellm.types.utils import ChatCompletionTokenLogprob, Choices, ModelResponse
from litellm.utils import token_counter
from openai import OpenAIError
from tokenizers import Tokenizer
from tqdm import tqdm

import polyflow
from polyflow.cache import CacheFactory
from polyflow.types import LMOutput, LMStats, LogprobsForCascade, LogprobsForFilterCascade

logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)


class LanguageProcessor:
    """
    A wrapper around language models that provides unified interface for text generation,
    token counting, and caching functionality.
    """
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        context_window: int = 128000,
        response_max_tokens: int = 512,
        concurrent_requests: int = 64,
        tokenizer_instance: Optional[Tokenizer] = None,
        cache_store=None,
        **model_params: Dict[str, Any],
    ):
        self.engine = model
        self.context_window = context_window
        self.response_max_tokens = response_max_tokens
        self.concurrent_requests = concurrent_requests
        self.tokenizer_instance = tokenizer_instance
        self.model_params = dict(temperature=temperature, max_tokens=response_max_tokens, **model_params)

        self.usage_statistics: LMStats = LMStats()
        self.cache_store = cache_store or CacheFactory.create_default_cache()

    def generate(
        self,
        message_lists: List[List[Dict[str, str]]],
        display_progress: bool = True,
        progress_message: str = "Processing uncached messages",
        **generation_params: Dict[str, Any],
    ) -> LMOutput:
        """
        Generate responses for multiple message lists, with caching and progress tracking.
        """
        run_params = {**self.model_params, **generation_params}

        # Configure logprobs if needed
        if run_params.get("logprobs", False):
            run_params.setdefault("top_logprobs", 10)

        # Check cache and identify which messages need processing
        message_hashes = [self._create_request_hash(msg, run_params) for msg in message_lists]
        cached_results = [self.cache_store.get(hash_id) for hash_id in message_hashes]
        pending_requests = [
            (msg, hash_id) for msg, hash_id, cache_result in zip(message_lists, message_hashes, cached_results) 
            if cache_result is None
        ]
        self.usage_statistics.total_usage.cache_hits += len(message_lists) - len(pending_requests)

        # Process messages that weren't in the cache
        new_responses = self._process_pending_requests(
            pending_requests, run_params, display_progress, progress_message
        )

        # Store new responses in cache
        for response, (_, hash_id) in zip(new_responses, pending_requests):
            self._store_in_cache(response, hash_id)

        # Combine all results in the original order
        complete_responses = self._combine_responses(cached_results, new_responses)
        response_texts = [self._extract_response_text(resp) for resp in complete_responses]
        
        token_probabilities = None
        if run_params.get("logprobs"):
            token_probabilities = [self._extract_token_probabilities(resp) for resp in complete_responses]

        return LMOutput(outputs=response_texts, logprobs=token_probabilities)

    def _process_pending_requests(self, pending_data, run_params, display_progress, progress_message):
        """Processes message batches that weren't found in cache."""
        pending_responses = []
        request_count = len(pending_data)

        with tqdm(
            total=request_count,
            desc=progress_message,
            disable=not display_progress,
            bar_format="{l_bar}{bar} {n}/{total} LM calls [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as progress_bar:
            for batch_start in range(0, request_count, self.concurrent_requests):
                batch_end = batch_start + self.concurrent_requests
                current_batch = [msg for msg, _ in pending_data[batch_start:batch_end]]
                batch_responses = batch_completion(self.engine, current_batch, drop_params=True, **run_params)
                pending_responses.extend(batch_responses)
                
                progress_bar.update(len(current_batch))

        return pending_responses

    def _store_in_cache(self, response, hash_id):
        """Stores a response in cache and updates usage statistics."""
        if isinstance(response, OpenAIError):
            raise response
            
        self._update_usage_statistics(response)
        self.cache_store.insert(hash_id, response)

    def _create_request_hash(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> str:
        """Creates a unique hash for a message and parameter combination."""
        hash_input = str(self.engine) + str(messages) + str(params)
        return hashlib.sha256(hash_input.encode()).hexdigest()

    def _combine_responses(
        self, cached_responses: List[Optional[ModelResponse]], new_responses: List[ModelResponse]
    ) -> List[ModelResponse]:
        """Combines cached and newly generated responses in the original order."""
        new_response_iter = iter(new_responses)
        return [response if response is not None else next(new_response_iter) for response in cached_responses]

    def _update_usage_statistics(self, response: ModelResponse):
        """Updates token usage statistics from a response."""
        if not hasattr(response, "usage"):
            return

        self.usage_statistics.total_usage.prompt_tokens += response.usage.prompt_tokens
        self.usage_statistics.total_usage.completion_tokens += response.usage.completion_tokens
        self.usage_statistics.total_usage.total_tokens += response.usage.total_tokens

        try:
            self.usage_statistics.total_usage.total_cost += completion_cost(completion_response=response)
        except litellm.exceptions.NotFoundError as e:
            # Sometimes the model's pricing information is not available
            polyflow.logger.debug(f"Error calculating completion cost: {e}")

    def _extract_response_text(self, response: ModelResponse) -> str:
        """Extracts the text content from a model response."""
        choice = response.choices[0]
        assert isinstance(choice, Choices)
        if choice.message.content is None:
            raise ValueError(f"Response contains no content: {response}")
        return choice.message.content

    def _extract_token_probabilities(self, response: ModelResponse) -> List[ChatCompletionTokenLogprob]:
        """Extracts token probabilities from a model response."""
        choice = response.choices[0]
        assert isinstance(choice, Choices)
        token_logprobs = choice.logprobs["content"]
        return [ChatCompletionTokenLogprob(**logprob) for logprob in token_logprobs]

    def prepare_cascade_logprobs(self, logprobs: List[List[ChatCompletionTokenLogprob]]) -> LogprobsForCascade:
        """Formats token logprobs for use in cascade operations."""
        all_tokens = []
        all_confidences = []
        
        for response_logprobs in logprobs:
            tokens = [token_data.token for token_data in response_logprobs]
            confidences = [np.exp(token_data.logprob) for token_data in response_logprobs]
            all_tokens.append(tokens)
            all_confidences.append(confidences)
            
        return LogprobsForCascade(tokens=all_tokens, confidences=all_confidences)

    def prepare_filter_cascade_logprobs(
        self, logprobs: List[List[ChatCompletionTokenLogprob]]
    ) -> LogprobsForFilterCascade:
        """Formats logprobs specifically for filter cascade operations."""
        # Get basic cascade format
        base_format = self.prepare_cascade_logprobs(logprobs)
        true_probabilities = []

        def calculate_true_probability(token_probs: Dict[str, float]) -> Optional[float]:
            """Calculate normalized probability for 'True' given True/False alternatives."""
            if "True" in token_probs and "False" in token_probs:
                true_value = token_probs["True"]
                false_value = token_probs["False"]
                return true_value / (true_value + false_value)
            return None

        # Extract true probabilities for filter operations
        for idx, response_logprobs in enumerate(logprobs):
            true_prob = None
            
            # Look for True/False token probabilities
            for token_data in response_logprobs:
                top_tokens_probs = {
                    alt_token.token: np.exp(alt_token.logprob) 
                    for alt_token in token_data.top_logprobs
                }
                true_prob = calculate_true_probability(top_tokens_probs)
                if true_prob is not None:
                    break

            # Default handling if True/False not found
            if true_prob is None:
                true_prob = 1 if "True" in base_format.tokens[idx] else 0

            true_probabilities.append(true_prob)

        return LogprobsForFilterCascade(
            tokens=base_format.tokens, 
            confidences=base_format.confidences, 
            true_probs=true_probabilities
        )

    def count_input_tokens(self, content: Union[List[Dict[str, str]], str]) -> int:
        """Counts tokens in a message or message list."""
        if isinstance(content, str):
            content = [{"role": "user", "content": content}]

        tokenizer_config = None
        if self.tokenizer_instance:
            tokenizer_config = dict(type="huggingface_tokenizer", tokenizer=self.tokenizer_instance)

        return token_counter(
            custom_tokenizer=tokenizer_config,
            model=self.engine,
            messages=content,
        )

    def display_usage_summary(self):
        """Displays a summary of token usage and costs."""
        print(f"Total cost: ${self.usage_statistics.total_usage.total_cost:.6f}")
        print(f"Total prompt tokens: {self.usage_statistics.total_usage.prompt_tokens}")
        print(f"Total completion tokens: {self.usage_statistics.total_usage.completion_tokens}")
        print(f"Total tokens: {self.usage_statistics.total_usage.total_tokens}")
        print(f"Total cache hits: {self.usage_statistics.total_usage.cache_hits}")

    def reset_usage_statistics(self):
        """Resets all usage statistics to zero."""
        self.usage_statistics = LMStats(
            total_usage=LMStats.TotalUsage(
                prompt_tokens=0, 
                completion_tokens=0, 
                total_tokens=0, 
                total_cost=0.0,
                cache_hits=0
            )
        )

    def clear_cache(self, new_max_size: Optional[int] = None):
        """Clears the response cache."""
        self.cache_store.reset(new_max_size)
        
    # Alias methods to maintain compatibility with existing code
    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)
        
    def print_total_usage(self):
        return self.display_usage_summary()
        
    def reset_stats(self):
        return self.reset_usage_statistics()
        
    def reset_cache(self, max_size=None):
        return self.clear_cache(max_size)
        
    def format_logprobs_for_cascade(self, logprobs):
        return self.prepare_cascade_logprobs(logprobs)
        
    def format_logprobs_for_filter_cascade(self, logprobs):
        return self.prepare_filter_cascade_logprobs(logprobs)
        
    def count_tokens(self, messages):
        return self.count_input_tokens(messages)

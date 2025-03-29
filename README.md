# PolyFlow: Advanced LLM-Powered Data Processing Framework

## Overview
PolyFlow is a sophisticated data processing framework that seamlessly integrates Language Models (LLMs) and vector embeddings with pandas DataFrame operations. It enables semantic analysis of both structured and unstructured data through an intuitive API.

## Installation

```bash
pip install polyflow-dev-ai
```

## Key Features

### Integrated Model Ecosystem
- **Language Models** via `LanguageProcessor` class
  - Support for OpenAI models (GPT-4o, GPT-3.5)
  - Claude/Anthropic models integration
  - Configurable parameters (temperature, max tokens)
  - Built-in token counting and caching

- **Vector Embeddings** via `SentenceVectorizer` class
  - Built on sentence-transformers with FAISS for similarity search
  - Support for modern multilingual models (e.g., E5, multilingual-E5)
  - Efficient batch processing and index management
  - Customizable similarity metrics

- **Specialized Models**
  - `TemporalLanguageProcessor`: Time series analysis and anomaly detection
  - `LlamaProcessor`: Local LLM inference with llama.cpp models
  - `EncoderReranker`: Cross-encoder based reranking

### Semantic DataFrame Operations
- **Vector Transformations**: `vector_transform()` to process text with embedding similarity
- **Semantic Joins**: `vector_join()` and `sem_sim_join()` to connect related data
- **Classification**: `sem_classify()` and `sem_hybrid_classify()` with LLM + embedding power
- **Clustering**: `sk_sem_cluster()` for semantically meaningful grouping
- **Time Series Analysis**: `vecsem_time_series` for pattern and anomaly detection

## Quick Start Examples

### Semantic Classification with Hybrid Approach
```python
import pandas as pd
import polyflow
from polyflow.models import LanguageProcessor, SentenceVectorizer

# Configure models
lm = LanguageProcessor(model="gpt-4o", api_key="YOUR_API_KEY")
rm = SentenceVectorizer(model="intfloat/multilingual-e5-large")
polyflow.settings.configure(lm=lm, rm=rm)

# Classify customer support tickets
tickets_df = pd.DataFrame({
    "description": [
        "Unable to login to my account after password reset",
        "App crashes when uploading large files",
        "Please add dark mode to the dashboard"
    ]
})

categories = ["Bug", "Feature Request", "Account Issue", "Feedback"]

# Use hybrid classification (LLM + embeddings)
results = tickets_df.sem_hybrid_classify(
    text_column="description",
    categories=categories,
    use_embeddings=True,
    return_scores=True,
    return_reasoning=True
)

print(results[["description", "classification", "confidence_score"]])
```

### Semantic Join Between Datasets
```python
# Connect courses with relevant skills
courses_df = pd.DataFrame({
    "Course Name": ["Machine Learning Fundamentals", "Database Systems Design"]
})

skills_df = pd.DataFrame({
    "Skill": ["Python Programming", "Data Analysis", "Database Administration"]
})

# Perform semantic join
results = courses_df.vector_join(
    skills_df, 
    "Analyze how directly studying {Course Name} develops competency in {Skill}"
)

print(results[["Course Name", "Skill", "score"]])
```

## Advanced Usage

### Time Series Analysis
```python
# Detect anomalies in time series data
from polyflow.models import TemporalLanguageProcessor

lm = TemporalLanguageProcessor(
    model_identifier="gpt-4o",
    api_credentials="YOUR_API_KEY",
)
polyflow.settings.configure(lm=lm)

# Analyze time series data
anomalies = time_series_df.vecsem_time_series.detect_anomalies(
    time_col="timestamp",
    value_col="value",
    description="Identify unusual patterns including spikes and drops"
)
```

### Local LLM Support
```python
# Use local Llama models
from polyflow.models import LlamaProcessor

llm = LlamaProcessor(
    model_path="path/to/model.gguf",
    generation_temperature=0.1
)

polyflow.settings.configure(lm=llm)
```

## Architecture

PolyFlow is designed as a modular framework with:
- A core settings system to manage model configurations
- Extension mechanisms for custom models and operations
- Flexible caching to optimize performance
- DataFrame-centric API for seamless integration with pandas workflows

## Project Status

This project is actively maintained and expanded with new capabilities. Contributions are welcome!

## License

MIT
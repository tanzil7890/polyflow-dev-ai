# PolyFlow Sheets

A Google Sheets-like web interface for PolyFlow that enables semantic analysis of tabular data with an intuitive UI.

## Overview

PolyFlow Sheets provides a spreadsheet-like interface for exploring and analyzing data using PolyFlow's powerful semantic operations. It combines the familiarity of a spreadsheet with advanced capabilities like:

- Semantic classification
- Vector-based transformations
- LLM-powered text analysis
- Semantic clustering
- Vector joins between datasets
- Time series analysis
- Interactive visualizations

## Features

- **Spreadsheet Interface**: Familiar row/column grid for data entry and viewing
- **Command Bar**: Execute PolyFlow operations with a simple command syntax
- **Visualization Panel**: View charts and graphs of your data
- **Example Datasets**: Pre-loaded examples to get started quickly
- **Command History**: Track and reuse previous commands

## Installation

1. Clone this repository and navigate to the project directory
2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key:
   ```
   export OPENAI_API_KEY=your_api_key
   ```

## Usage

1. Start the application:
   ```
   python app.py
   ```
2. Open your browser and go to `http://localhost:5000`
3. Create a new sheet or load an example dataset
4. Enter data into cells or use the example datasets
5. Execute PolyFlow commands in the command bar

## Command Examples

### Vector Transformations
```
vector_transform(column="description", transform_query="Extract sentiment", threshold=0.7)
```

### Semantic Classification
```
sem_classify(text_column="feedback", categories=["Positive", "Negative", "Neutral"])
```

### Hybrid Classification
```
sem_hybrid_classify(text_column="description", categories=["Bug", "Feature Request", "Question"], return_scores=true)
```

### LLM Transformation
```
llm_transform(text_column="description", instruction="Summarize in 5 words")
```

### Semantic Clustering
```
sk_sem_cluster(text_column="description", n_clusters=3)
```

### Vector Join
```
vector_join(column="product_name", join_df="Categories", join_column="category", user_instruction="Does this product belong in this category?")
```

### Time Series Analysis
```
vecsem_time_series(time_col="date", value_col="sales", operation="detect_anomalies")
```

### Visualizations
```
visualize(type="bar", x="category", y="count", title="Categories Distribution")
visualize(type="line", x="date", y="sales")
visualize(type="scatter", x="x_embed", y="y_embed")
visualize(type="pie", x="category", y="sales")
visualize(type="heatmap", x="product", y="sales", pivot="region")
```

## Requirements

- Python 3.8+
- Flask
- Pandas
- Plotly
- NumPy
- Scikit-learn
- PolyFlow library

## License

MIT

## Acknowledgements

This project uses the PolyFlow framework to provide semantic analysis capabilities. 
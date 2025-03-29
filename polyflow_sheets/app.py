import os
import sys
import json
import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Add the parent directory to sys.path to import local polyflow package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polyflow
from polyflow.models import LanguageProcessor, SentenceVectorizer
from polyflow_sheets.data_loader import DataLoader

app = Flask(__name__)
app.config['SECRET_KEY'] = 'polyflow-sheets-secret-key'

# Global variables for data storage
sheets = {}
current_sheet = None

# Configure polyflow models
def configure_polyflow():
    """Configure polyflow models with appropriate API keys"""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables")
        api_key = "dummy_key"  # For testing without actual API calls
    
    # Initialize models
    lm = LanguageProcessor(model="gpt-4o", api_key=api_key)
    rm = SentenceVectorizer(model_name="intfloat/multilingual-e5-large")
    
    # Configure settings
    polyflow.settings.configure(lm=lm, rm=rm)
    return "Models configured successfully"

@app.route('/')
def index():
    """Render the main application page"""
    return render_template('index.html', sheets=sheets, current_sheet=current_sheet, 
                          example_datasets=DataLoader.get_available_datasets())

@app.route('/create_sheet', methods=['POST'])
def create_sheet():
    """Create a new sheet"""
    sheet_name = request.form.get('sheet_name', 'Sheet1')
    data = request.form.get('data', '{}')
    
    try:
        data = json.loads(data)
    except json.JSONDecodeError:
        data = {}
    
    # Create a new DataFrame
    columns = data.get('columns', ['A', 'B', 'C', 'D', 'E'])
    rows = data.get('rows', 10)
    df = pd.DataFrame(None, index=range(rows), columns=columns)
    
    # Store the DataFrame
    sheets[sheet_name] = df
    global current_sheet
    current_sheet = sheet_name
    
    return redirect(url_for('index'))

@app.route('/load_example_dataset', methods=['POST'])
def load_example_dataset():
    """Load an example dataset"""
    dataset_name = request.form.get('dataset_name')
    sheet_name = request.form.get('sheet_name', dataset_name.replace('_', ' ').title())
    
    try:
        # Load the dataset
        df = DataLoader.load_dataset(dataset_name)
        
        # Store the DataFrame
        sheets[sheet_name] = df
        global current_sheet
        current_sheet = sheet_name
        
        return jsonify({
            'success': True,
            'message': f'Dataset "{dataset_name}" loaded successfully'
        })
    except Exception as e:
        return jsonify({
            'error': f'Error loading dataset: {str(e)}'
        })

@app.route('/get_sheet_data')
def get_sheet_data():
    """Return the current sheet data as JSON"""
    sheet_name = request.args.get('sheet_name', current_sheet)
    if sheet_name not in sheets:
        return jsonify({'error': 'Sheet not found'})
    
    df = sheets[sheet_name]
    return jsonify({
        'columns': df.columns.tolist(),
        'data': df.fillna('').to_dict(orient='records')
    })

@app.route('/update_cell', methods=['POST'])
def update_cell():
    """Update a cell value"""
    sheet_name = request.form.get('sheet_name', current_sheet)
    row = int(request.form.get('row'))
    col = request.form.get('col')
    value = request.form.get('value')
    
    if sheet_name not in sheets:
        return jsonify({'error': 'Sheet not found'})
    
    # Update the cell
    sheets[sheet_name].at[row, col] = value
    return jsonify({'success': True})

@app.route('/run_command', methods=['POST'])
def run_command():
    """Execute a polyflow command on the current sheet"""
    sheet_name = request.form.get('sheet_name', current_sheet)
    command = request.form.get('command', '')
    
    if sheet_name not in sheets:
        return jsonify({'error': 'Sheet not found'})
    
    df = sheets[sheet_name]
    result = None
    error = None
    
    try:
        # Parse and execute the command
        if command.startswith('vector_transform'):
            # Example: vector_transform(column="A", transform_query="Extract keywords")
            params = parse_command_params(command, 'vector_transform')
            result = df.vector_transform(**params)
            sheets[sheet_name] = result
        
        elif command.startswith('sem_classify'):
            # Example: sem_classify(text_column="A", categories=["Category1", "Category2"])
            params = parse_command_params(command, 'sem_classify')
            if 'categories' in params and isinstance(params['categories'], str):
                # Convert string of categories to list
                params['categories'] = eval(params['categories'])
            result = df.sem_classify(**params)
            sheets[sheet_name] = result
        
        elif command.startswith('sem_hybrid_classify'):
            # Handle hybrid classification
            params = parse_command_params(command, 'sem_hybrid_classify')
            if 'categories' in params and isinstance(params['categories'], str):
                params['categories'] = eval(params['categories'])
            result = df.sem_hybrid_classify(**params)
            sheets[sheet_name] = result
        
        elif command.startswith('sk_sem_cluster'):
            # Example: sk_sem_cluster(text_column="A", n_clusters=3)
            params = parse_command_params(command, 'sk_sem_cluster')
            try:
                result = df.sk_sem_cluster(**params)
                sheets[sheet_name] = result
            except ValueError as e:
                # Handle specific clustering errors with user-friendly messages
                if "Insufficient data for clustering" in str(e):
                    error_msg = str(e)
                    # Extract numbers for better error message
                    import re
                    matches = re.findall(r'\d+', error_msg)
                    if len(matches) >= 3:
                        current, required, clusters = matches[0], matches[1], matches[2]
                        error = f"Cannot cluster {current} data points into {clusters} clusters. Need at least {required} data points. Try reducing the number of clusters or adding more data."
                    else:
                        error = str(e)
                else:
                    error = str(e)
                return jsonify({'error': error})
            except Exception as e:
                error = f"Clustering error: {str(e)}. This might be due to insufficient data points for the requested number of clusters."
                return jsonify({'error': error})
        
        elif command.startswith('llm_transform'):
            # Example: llm_transform(text_column="A", instruction="Summarize this text")
            params = parse_command_params(command, 'llm_transform')
            result = df.llm_transform(**params)
            sheets[sheet_name] = result
        
        elif command.startswith('vector_join'):
            # Example: vector_join(column="A", join_df="Sheet2", join_column="B", user_instruction="Compare these items")
            params = parse_command_params(command, 'vector_join')
            
            # Check if join_df parameter is provided and valid
            if 'join_df' not in params:
                raise ValueError("join_df parameter is required for vector_join")
            
            join_sheet_name = params.pop('join_df')
            if join_sheet_name not in sheets:
                raise ValueError(f"Sheet '{join_sheet_name}' not found")
            
            join_df = sheets[join_sheet_name]
            join_column = params.pop('join_column', None)
            
            if not join_column:
                raise ValueError("join_column parameter is required for vector_join")
            
            # Perform the join operation
            result = perform_vector_join(df, join_df, params['column'], join_column, 
                                        params.get('user_instruction', 'Compare these items for similarity'))
            
            sheets[sheet_name] = result
        
        elif command.startswith('vecsem_time_series'):
            # Example: vecsem_time_series(time_col="date", value_col="value", operation="detect_anomalies")
            params = parse_command_params(command, 'vecsem_time_series')
            
            operation = params.pop('operation', 'detect_anomalies')
            time_col = params.pop('time_col', None)
            value_col = params.pop('value_col', None)
            
            if not time_col or not value_col:
                raise ValueError("time_col and value_col parameters are required for vecsem_time_series")
            
            # Perform time series analysis
            if operation == 'detect_anomalies':
                result = df.vecsem_time_series.detect_anomalies(time_col=time_col, value_col=value_col, **params)
            elif operation == 'forecast':
                result = df.vecsem_time_series.forecast(time_col=time_col, value_col=value_col, **params)
            else:
                raise ValueError(f"Unknown time series operation: {operation}")
                
            sheets[sheet_name] = result
            
        elif command.startswith('visualize'):
            # Special command to create visualizations
            return create_visualization(df, command)
        
        else:
            error = f"Unknown command: {command}"
    
    except Exception as e:
        error = f"Error executing command: {str(e)}"
    
    if error:
        return jsonify({'error': error})
    
    return jsonify({
        'success': True,
        'columns': result.columns.tolist(),
        'data': result.fillna('').to_dict(orient='records')
    })

def perform_vector_join(df1, df2, col1, col2, instruction):
    """Helper function to perform vector join between two dataframes"""
    # Extract series from dataframes
    series1 = df1[col1]
    series2 = df2[col2]
    
    # Create ids for joining
    ids1 = list(range(len(series1)))
    ids2 = list(range(len(series2)))
    
    # Perform the join
    join_result = polyflow.vector_join(
        series1, series2, 
        ids1, ids2, 
        col1, col2, 
        polyflow.settings.lm,
        instruction
    )
    
    # Create a result dataframe with the joined data
    result_data = []
    for id1, id2, explanation in join_result.join_results:
        row = {
            f"{col1}": series1.iloc[id1],
            f"{col2}": series2.iloc[id2],
            "similarity_score": 1.0,  # Placeholder, could calculate actual similarity
            "explanation": explanation or "Items are similar"
        }
        result_data.append(row)
    
    # Return as dataframe
    return pd.DataFrame(result_data)

def parse_command_params(command, prefix):
    """Parse command parameters from a string"""
    # Extract the parameters part
    params_str = command.replace(f"{prefix}(", "").rstrip(")")
    
    # Convert to a dictionary using eval
    # This is a simplified approach, a more robust parser would be better in production
    params_dict = {}
    pairs = params_str.split(",")
    
    for pair in pairs:
        if "=" in pair:
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()
            
            # Handle quoted strings
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            elif value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '', 1).isdigit():
                value = float(value)
            
            params_dict[key] = value
    
    return params_dict

def create_visualization(df, command):
    """Create a visualization based on the command"""
    params = parse_command_params(command, 'visualize')
    
    viz_type = params.get('type', 'bar')
    x_col = params.get('x')
    y_col = params.get('y')
    
    if not x_col or (viz_type != 'pie' and not y_col):
        return jsonify({'error': 'Missing required columns for visualization'})
    
    try:
        if viz_type == 'bar':
            fig = px.bar(df, x=x_col, y=y_col, title=params.get('title', 'Bar Chart'))
        elif viz_type == 'line':
            fig = px.line(df, x=x_col, y=y_col, title=params.get('title', 'Line Chart'))
        elif viz_type == 'scatter':
            fig = px.scatter(df, x=x_col, y=y_col, title=params.get('title', 'Scatter Plot'))
        elif viz_type == 'pie':
            fig = px.pie(df, names=x_col, values=y_col, title=params.get('title', 'Pie Chart'))
        elif viz_type == 'heatmap':
            # For heatmap, we need to pivot the data
            pivot_col = params.get('pivot')
            if not pivot_col:
                return jsonify({'error': 'Missing pivot column for heatmap'})
            pivot_df = df.pivot(index=x_col, columns=pivot_col, values=y_col)
            fig = px.imshow(pivot_df, title=params.get('title', 'Heatmap'))
        else:
            return jsonify({'error': f'Unsupported visualization type: {viz_type}'})
        
        # Convert to HTML
        plot_html = pio.to_html(fig, full_html=False)
        return jsonify({
            'success': True,
            'visualization': plot_html
        })
    
    except Exception as e:
        return jsonify({'error': f'Error creating visualization: {str(e)}'})

@app.route('/get_example_datasets')
def get_example_datasets():
    """Return a list of available example datasets"""
    return jsonify({
        'datasets': DataLoader.get_available_datasets()
    })

if __name__ == '__main__':
    # Configure polyflow if running directly
    configure_polyflow()
    app.run(debug=True, host='0.0.0.0', port=8000) 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataLoader:
    """Utility class to load example datasets for PolyFlow Sheets"""
    
    @staticmethod
    def load_customer_feedback():
        """Load a customer feedback dataset"""
        data = {
            "customer_id": [f"CUST{i:04d}" for i in range(1, 11)],
            "feedback": [
                "The product is amazing! I love how intuitive it is to use.",
                "Had some issues with installation, but customer support helped me right away.",
                "Not worth the price. There are better alternatives available.",
                "This tool has completely transformed how I work. Highly recommend!",
                "The app crashes frequently when I try to export large files.",
                "Decent product, but missing some key features I need.",
                "Customer service was unhelpful and slow to respond.",
                "Great value for money, does everything I need and more.",
                "Love the recent updates, the new dashboard feature is exactly what I was looking for!",
                "Complicated interface with a steep learning curve."
            ],
            "rating": [5, 4, 2, 5, 2, 3, 1, 5, 5, 3],
            "submission_date": [
                (datetime.now() - timedelta(days=i*3)).strftime("%Y-%m-%d") 
                for i in range(10)
            ],
            "product": [
                "Premium Suite", "Basic Package", "Premium Suite", "Basic Package",
                "Enterprise Solution", "Premium Suite", "Basic Package", "Enterprise Solution",
                "Premium Suite", "Enterprise Solution"
            ]
        }
        return pd.DataFrame(data)
    
    @staticmethod
    def load_product_descriptions():
        """Load a product description dataset"""
        data = {
            "product_id": [f"PROD{i:03d}" for i in range(1, 8)],
            "name": [
                "Advanced Analytics Suite", 
                "Data Integration Platform", 
                "Visualization Toolkit",
                "Natural Language Processor",
                "Time Series Predictor",
                "Document Understanding System",
                "Entity Resolution Engine"
            ],
            "description": [
                "A comprehensive suite of advanced analytics tools for data scientists and analysts to build, test, and deploy machine learning models.",
                "Connect to any data source and transform your data with this powerful ETL and data integration platform.",
                "Create stunning interactive visualizations and dashboards from your data with minimal coding.",
                "Process and understand text data with this natural language processing toolkit, featuring sentiment analysis and entity extraction.",
                "Predict future values and detect anomalies in time series data using advanced statistical models and machine learning.",
                "Extract structured information from unstructured documents including PDFs, images, and scanned papers.",
                "Match and merge records across databases to create a unified view of entities like customers or products."
            ],
            "category": [
                "Analytics", "Data Engineering", "Visualization", 
                "NLP", "Forecasting", "Document Processing", "Data Quality"
            ],
            "price_tier": ["Enterprise", "Standard", "Standard", "Enterprise", "Standard", "Premium", "Premium"]
        }
        return pd.DataFrame(data)
    
    @staticmethod
    def load_sales_data():
        """Load a sales dataset"""
        np.random.seed(42)  # For reproducibility
        
        regions = ["North", "South", "East", "West", "Central"]
        products = ["Software", "Hardware", "Services", "Support", "Training"]
        
        # Generate dates for the last 12 months
        dates = [(datetime.now() - timedelta(days=30*i)).strftime("%Y-%m") for i in range(12)]
        
        # Create dataset
        data = []
        for region in regions:
            for product in products:
                for date in dates:
                    base_value = np.random.randint(5000, 15000)
                    # Add some product-specific trends
                    if product == "Software":
                        base_value *= 1.5  # Software sells better
                    if product == "Training" and region in ["North", "Central"]:
                        base_value *= 0.8  # Training sells worse in these regions
                    
                    # Add time trend (recent months have higher sales)
                    month_idx = dates.index(date)
                    trend_factor = 1 + (11 - month_idx) * 0.02
                    
                    sales = int(base_value * trend_factor * (0.9 + np.random.random() * 0.3))
                    
                    data.append({
                        "date": date,
                        "region": region,
                        "product": product,
                        "sales": sales,
                        "units": int(sales / np.random.randint(500, 1000))
                    })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def load_support_tickets():
        """Load a support tickets dataset"""
        data = {
            "ticket_id": [f"TCKT{i:05d}" for i in range(1, 16)],
            "description": [
                "Unable to log in to my account after password reset",
                "Application crashes when uploading files larger than 10MB",
                "Need help exporting data to CSV format",
                "Error message 'Server connection lost' appears randomly",
                "Dashboard widgets not loading properly",
                "Feature request: Add dark mode to the interface",
                "API integration with Salesforce not working",
                "Need assistance with configuring email notifications",
                "Mobile app keeps freezing on data sync",
                "Can't find the export function mentioned in documentation",
                "Users unable to access shared reports",
                "Performance degradation when filtering large datasets",
                "Request for additional user licenses",
                "Need training materials for new team members",
                "Integration with our custom CRM system"
            ],
            "priority": [
                "High", "Critical", "Low", "Medium", "Medium", 
                "Low", "High", "Medium", "High", "Low",
                "Critical", "High", "Low", "Medium", "Medium"
            ],
            "status": [
                "Open", "In Progress", "Closed", "Open", "Closed", 
                "Open", "In Progress", "Closed", "Open", "Closed",
                "In Progress", "Open", "Closed", "Closed", "Open"
            ],
            "created_date": [
                (datetime.now() - timedelta(days=i*2)).strftime("%Y-%m-%d") 
                for i in range(15)
            ]
        }
        return pd.DataFrame(data)
    
    @staticmethod
    def get_available_datasets():
        """Return a list of available example datasets"""
        return [
            "customer_feedback",
            "product_descriptions",
            "sales_data",
            "support_tickets"
        ]
    
    @staticmethod
    def load_dataset(name):
        """Load a dataset by name"""
        loaders = {
            "customer_feedback": DataLoader.load_customer_feedback,
            "product_descriptions": DataLoader.load_product_descriptions,
            "sales_data": DataLoader.load_sales_data,
            "support_tickets": DataLoader.load_support_tickets
        }
        
        if name not in loaders:
            raise ValueError(f"Dataset '{name}' not found. Available datasets: {DataLoader.get_available_datasets()}")
        
        return loaders[name]() 
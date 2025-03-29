import pandas as pd
import polyflow
from polyflow.models import LanguageProcessor, SentenceVectorizer

""" 
This is an example of how to use the sem_hybrid_classify method with a custom prompt or without a custom prompt.
"""

# Configure models
lm = LanguageProcessor(model="gpt-4o", api_key="OPEN_AI_API_KEY")
rm = SentenceVectorizer(model="intfloat/multilingual-e5-large")
polyflow.settings.configure(lm=lm, rm=rm)

# Example data: Customer Support Ticket Classification
tickets_data = {
    "ticket_id": range(1, 8),
    "description": [
        "Unable to login to my account after password reset",
        "App keeps crashing when uploading large files",
        "Thank you for the quick resolution to my billing issue",
        "Feature request: Add dark mode to the dashboard",
        "System error when processing payments",
        "Great customer service experience!",
        "Please add support for multiple languages"
    ]
}

df = pd.DataFrame(tickets_data)
categories = ["Bug", "Feature Request", "Account Issue", "Feedback"]

def print_analysis(results_df, title):
    """Helper function to print analysis results"""
    print(f"\n{title}")
    print("=" * 80)
    
    # Detailed results
    for _, row in results_df.iterrows():
        print(f"\nTicket ID: {row['ticket_id']}")
        print(f"Description: {row['description']}")
        print(f"Category: {row['classification']}")
        print(f"LM Confidence: {row['confidence_score']:.2f}")
        print(f"Embedding Similarity: {row['embedding_similarity']:.2f}")
        print(f"Model Agreement Score: {row['model_agreement']:.2f}")
        print(f"Reasoning: {row['classification_reasoning']}")
        print("-" * 40)

    # Category statistics
    category_stats = results_df.groupby('classification').agg({
        'confidence_score': ['mean', 'count'],
        'embedding_similarity': 'mean',
        'model_agreement': 'mean'
    }).round(2)
    
    print("\nCategory Statistics:")
    print(category_stats)
    
    # High confidence classifications
    high_confidence = results_df[
        (results_df['confidence_score'] > 0.8) & 
        (results_df['model_agreement'] == 1.0)
    ]
    print("\nHigh Confidence Classifications:")
    for _, row in high_confidence.iterrows():
        print(f"- {row['description']} -> {row['classification']} (Confidence: {row['confidence_score']:.2f})")
    
    # Model disagreements
    disagreements = results_df[results_df['model_agreement'] < 1.0]
    print("\nModel Disagreements:")
    for _, row in disagreements.iterrows():
        print(f"- {row['description']}")
        print(f"  LM Category: {row['classification']} (Confidence: {row['confidence_score']:.2f})")
        print(f"  Embedding Similarity: {row['embedding_similarity']:.2f}")

# 1. Default Classification (Basic Usage)
print("\n=== Basic Usage (Default Prompt) ===")
default_results = df.np_sem_hybrid_classify(
    text_column="description",
    categories=categories,
    use_embeddings=True,
    threshold=0.6,
    return_scores=True,
    return_reasoning=True
)
print_analysis(default_results, "Default Prompt Classification Results")

# 2. Advanced Usage with Custom Prompt
print("\n=== Advanced Usage (Custom Prompt) ===")
custom_prompt = (
    "As an expert support ticket classifier, analyze this ticket and categorize it into exactly one of "
    f"these categories: {', '.join(categories)}.\n\n"
    "Text: {text}\n\n"
    "Respond with a JSON object containing:\n"
    "- category: exactly one of the provided categories\n"
    "- confidence: number between 0 and 1 indicating classification confidence\n"
    "- reasoning: detailed explanation for the classification\n\n"
    "Consider:\n"
    "- Technical issues -> Bug\n"
    "- New functionality requests -> Feature Request\n"
    "- Login/user access -> Account Issue\n"
    "- Customer opinions/thanks -> Feedback"
)

custom_results = df.np_sem_hybrid_classify(
    text_column="description",
    categories=categories,
    prompt_template=custom_prompt,
    use_embeddings=True,
    threshold=0.6,
    return_scores=True,
    return_reasoning=True
)
print_analysis(custom_results, "Custom Prompt Classification Results")

# Compare results
print("\n=== Classification Comparison ===")
print("=" * 80)
comparison_df = pd.DataFrame({
    'Text': df['description'],
    'Default': default_results['classification'],
    'Custom': custom_results['classification'],
    'Default_Confidence': default_results['confidence_score'],
    'Custom_Confidence': custom_results['confidence_score'],
    'Agreement': default_results['classification'] == custom_results['classification']
})
print("\nClassification Differences:")
differences = comparison_df[~comparison_df['Agreement']]
for _, row in differences.iterrows():
    print(f"\nText: {row['Text']}")
    print(f"Default: {row['Default']} (Confidence: {row['Default_Confidence']:.2f})")
    print(f"Custom: {row['Custom']} (Confidence: {row['Custom_Confidence']:.2f})")
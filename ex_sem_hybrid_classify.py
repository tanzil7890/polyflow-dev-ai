import pandas as pd
import polyflow
from polyflow.models import LanguageProcessor, SentenceVectorizer

# Configure models
lm = LanguageProcessor(model="gpt-4o", api_key="OPEN_AI_API_KEY")
rm = SentenceVectorizer(model="intfloat/multilingual-e5-large")
polyflow.settings.configure(lm=lm, rm=rm)

# Example 1: Customer Support Ticket Classification
tickets_data = {
    "ticket_id": range(1, 12),
    "description": [
        "Unable to login to my account after resetting my password through the email link",
        "App repeatedly crashes when trying to upload files larger than 10MB to the cloud storage",
        "Thank you for the quick resolution to my billing issue. Your support team was excellent!",
        "Feature request: Please add dark mode option to the dashboard for reduced eye strain",
        "System error code E-5062 appears when processing credit card payments through the mobile app",
        "Great customer service experience! The representative was knowledgeable and solved my issue quickly",
        "Please add support for multiple languages in the user interface, especially Spanish and French",
        "The new update broke the search functionality - cannot find any of my previous documents",
        "Would like to request a refund for my subscription due to frequent service outages",
        "Need help with exporting my data in CSV format for use in our analytics platform",
        "The notification settings keep resetting after each app update, very frustrating experience"
    ]
}

df = pd.DataFrame(tickets_data)

# Define categories and custom prompt
categories = ["Bug", "Feature Request", "Account Issue", "Feedback", "Billing Issue"]
custom_prompt = (
    "As an expert support ticket classifier, analyze this ticket and categorize it into exactly one of "
    f"these categories: {', '.join(categories)}.\n\n"
    "Text: {text}\n\n"
    "Respond with a JSON object containing:\n"
    "- category: the most appropriate category\n"
    "- confidence: your confidence score (0-1)\n"
    "- reasoning: brief explanation for your choice"
)

# Classify tickets with both LM and embedding similarity
classified_tickets = df.sem_hybrid_classify(
    text_column="description",
    categories=categories,
    prompt_template=custom_prompt,
    use_embeddings=True,
    threshold=0.65,
    return_scores=True,
    return_reasoning=True
)

# Print detailed results
print("\nAdvanced Ticket Classification Results:")
print("=" * 80)
for _, row in classified_tickets.iterrows():
    print(f"\nTicket ID: {row['ticket_id']}")
    print(f"Description: {row['description']}")
    print(f"Category: {row['classification']}")
    print(f"LM Confidence: {row['confidence_score']:.2f}")
    print(f"Embedding Similarity: {row['embedding_similarity']:.2f}")
    print(f"Model Agreement Score: {row['model_agreement']:.2f}")
    print(f"Reasoning: {row['classification_reasoning']}")
    print("-" * 40)

# Example 2: Analysis of Classification Results
print("\nClassification Analysis:")
print("=" * 80)

# Calculate average confidence by category
category_stats = classified_tickets.groupby('classification').agg({
    'confidence_score': ['mean', 'count'],
    'embedding_similarity': 'mean',
    'model_agreement': 'mean'
}).round(2)

print("\nCategory Statistics:")
print(category_stats)

# Find high-confidence classifications
high_confidence = classified_tickets[
    (classified_tickets['confidence_score'] > 0.85) & 
    (classified_tickets['model_agreement'] == 1.0)
]
print("\nHigh Confidence Classifications:")
for _, row in high_confidence.iterrows():
    print(f"- {row['description']} -> {row['classification']} (Confidence: {row['confidence_score']:.2f})")

# Find cases with model disagreement
disagreements = classified_tickets[classified_tickets['model_agreement'] < 1.0]
print("\nModel Disagreements:")
for _, row in disagreements.iterrows():
    print(f"- {row['description']}")
    print(f"  LM Category: {row['classification']} (Confidence: {row['confidence_score']:.2f})")
    print(f"  Embedding Similarity: {row['embedding_similarity']:.2f}")


""" 
Advanced Ticket Classification Results:
================================================================================

Ticket ID: 1
Description: Unable to login to my account after resetting my password through the email link
Category: Account Issue
LM Confidence: 0.95
Embedding Similarity: 0.82
Model Agreement Score: 1.00
Reasoning: The ticket describes a problem with logging into an account after attempting a password reset, which is typically classified as an account-related issue.
----------------------------------------

Ticket ID: 2
Description: App repeatedly crashes when trying to upload files larger than 10MB to the cloud storage
Category: Bug
LM Confidence: 0.76
Embedding Similarity: 0.80
Model Agreement Score: 0.50
Reasoning: The ticket describes a functionality issue where the app crashes during a specific operation (uploading large files), which is indicative of a bug.
----------------------------------------

Ticket ID: 3
Description: Thank you for the quick resolution to my billing issue. Your support team was excellent!
Category: Feedback
LM Confidence: 0.76
Embedding Similarity: 0.83
Model Agreement Score: 0.50
Reasoning: The text expresses gratitude for resolving an issue, indicating it is feedback about a past interaction, specifically related to billing.
----------------------------------------

Ticket ID: 4
Description: Feature request: Please add dark mode option to the dashboard for reduced eye strain
Category: Feature Request
LM Confidence: 0.98
Embedding Similarity: 0.86
Model Agreement Score: 1.00
Reasoning: The text explicitly requests a new feature, specifically the addition of a dark mode to the dashboard.
----------------------------------------

Ticket ID: 5
Description: System error code E-5062 appears when processing credit card payments through the mobile app
Category: Bug
LM Confidence: 0.76
Embedding Similarity: 0.85
Model Agreement Score: 0.50
Reasoning: The text indicates a system error, which typically falls under a bug as it pertains to a malfunction in the system's operations.
----------------------------------------

Ticket ID: 6
Description: Great customer service experience! The representative was knowledgeable and solved my issue quickly
Category: Feedback
LM Confidence: 0.76
Embedding Similarity: 0.77
Model Agreement Score: 0.50
Reasoning: The text explicitly praises the customer service, indicating it is feedback rather than a bug, feature request, or account issue.
----------------------------------------

Ticket ID: 7
Description: Please add support for multiple languages in the user interface, especially Spanish and French
Category: Feature Request
LM Confidence: 0.95
Embedding Similarity: 0.85
Model Agreement Score: 1.00
Reasoning: The text explicitly requests the addition of new functionality (support for multiple languages), which is characteristic of a feature request.
----------------------------------------

Ticket ID: 8
Description: The new update broke the search functionality - cannot find any of my previous documents
Category: Bug
LM Confidence: 0.95
Embedding Similarity: 0.82
Model Agreement Score: 1.00
Reasoning: The text indicates a bug, as the search functionality is broken after an update, which is indicative of a malfunction in the system's operations.
----------------------------------------

Ticket ID: 9
Description: Would like to request a refund for my subscription due to frequent service outages
Category: Billing Issue
LM Confidence: 0.95
Embedding Similarity: 0.82
Model Agreement Score: 1.00
Reasoning: The text expresses a desire for a refund due to frequent service outages, which is indicative of a billing issue.
----------------------------------------

Ticket ID: 10
Description: Need help with exporting my data in CSV format for use in our analytics platform
Category: Feature Request
LM Confidence: 0.95
Embedding Similarity: 0.82
Model Agreement Score: 1.00
Reasoning: The text expresses a need for a feature, specifically the export of data in CSV format for use in an analytics platform, which is characteristic of a feature request.
----------------------------------------

Ticket ID: 11
Description: The notification settings keep resetting after each app update, very frustrating experience
Category: Bug
LM Confidence: 0.95
Embedding Similarity: 0.82
Model Agreement Score: 1.00
Reasoning: The text indicates a bug, as the notification settings are resetting after each app update, which is indicative of a malfunction in the system's operations.
----------------------------------------

Classification Analysis:
================================================================================

Category Statistics:
                confidence_score       embedding_similarity model_agreement
                            mean count                 mean            mean
classification                                                             
Account Issue               0.95     1                 0.82             1.0
Bug                         0.76     2                 0.82             0.5
Feature Request             0.96     2                 0.86             1.0
Feedback                    0.76     2                 0.80             0.5
Billing Issue               0.95     1                 0.82             1.0

High Confidence Classifications:
- Unable to login to my account after resetting my password through the email link -> Account Issue (Confidence: 0.95)
- Feature request: Please add dark mode option to the dashboard for reduced eye strain -> Feature Request (Confidence: 0.98)
- Please add support for multiple languages in the user interface, especially Spanish and French -> Feature Request (Confidence: 0.95)

Model Disagreements:
- App repeatedly crashes when trying to upload files larger than 10MB to the cloud storage
  LM Category: Bug (Confidence: 0.76)
  Embedding Similarity: 0.80
- Thank you for the quick resolution to my billing issue. Your support team was excellent!
  LM Category: Feedback (Confidence: 0.76)
  Embedding Similarity: 0.83
- System error code E-5062 appears when processing credit card payments through the mobile app
  LM Category: Bug (Confidence: 0.76)
  Embedding Similarity: 0.85
- Great customer service experience! The representative was knowledgeable and solved my issue quickly
  LM Category: Feedback (Confidence: 0.76)
  Embedding Similarity: 0.77
- The notification settings keep resetting after each app update, very frustrating experience
  LM Category: Bug (Confidence: 0.95)
  Embedding Similarity: 0.82


 """
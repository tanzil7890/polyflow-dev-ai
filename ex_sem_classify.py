import pandas as pd
import polyflow
from polyflow.models import LanguageProcessor, SentenceVectorizer

def setup_api_credentials():
    """Setup OpenAI API credentials"""
    api_key = "OPENAI_API_KEY"
    
    return api_key

# Configure models
lm = LanguageProcessor(model="gpt-4o", api_key=setup_api_credentials())
rm = SentenceVectorizer(model="intfloat/multilingual-e5-large")
polyflow.settings.configure(lm=lm, rm=rm)

# Sample smartphone feature reviews dataset
reviews_data = {
    "product_id": range(1, 11),
    "review_text": [
        "The battery life is exceptional, easily lasting two full days of heavy usage without needing a charge",
        "Battery drains extremely fast, even in standby mode. Can't make it through a workday without charging twice",
        "This phone's camera system is outstanding - the detail in low-light photos is unlike anything I've seen before",
        "Camera quality is disappointing, photos look grainy even in good lighting conditions",
        "The facial recognition unlocks the phone instantly, even in complete darkness",
        "The fingerprint sensor works about 30% of the time, extremely unreliable and frustrating",
        "Water resistance worked perfectly when I accidentally dropped it in the pool",
        "Despite claims of water resistance, it stopped working after light rain exposure",
        "The display refresh rate makes scrolling and gaming incredibly smooth",
        "Interface is laggy and unresponsive, makes using basic apps frustrating"
    ]
}

df = pd.DataFrame(reviews_data)

# Classify reviews into sentiment categories with more detailed reasoning
classified_reviews = df.sem_classify(
    text_column="review_text",
    categories=["Very Positive", "Positive", "Neutral", "Negative", "Very Negative"],
    return_scores=True,
    return_reasoning=True
)

print("\nSmartphone Feature Review Classification:")
print("=" * 80)
for _, row in classified_reviews.iterrows():
    print(f"\nReview: {row['review_text']}")
    print(f"Classification: {row['classification']}")
    print(f"Confidence: {row['confidence_score']:.2f}")
    print(f"Reasoning: {row['classification_reasoning']}")

# Analyze classification distribution
sentiment_distribution = classified_reviews['classification'].value_counts()
print("\nSentiment Distribution:")
print(sentiment_distribution)

# Find reviews with highest confidence scores
highest_confidence = classified_reviews.sort_values('confidence_score', ascending=False).head(3)
print("\nHighest Confidence Classifications:")
for _, row in highest_confidence.iterrows():
    print(f"- {row['review_text']} -> {row['classification']} (Confidence: {row['confidence_score']:.2f})")


""" 
Classification Results with confidence and reasoning:

Review: The battery life is terrible, doesn't last more than 2 hours
Classification: Negative
Confidence: 0.95
Reasoning: The text expresses dissatisfaction with the battery life, indicating it is 'terrible' and does not last long, which clearly conveys a negative sentiment.

Review: Battery drains too quickly, very disappointing
Classification: Negative
Confidence: 0.95
Reasoning: The text expresses dissatisfaction with the battery performance, indicating disappointment, which aligns with a negative sentiment.

Review: Amazing camera quality, photos are crystal clear
Classification: Positive
Confidence: 0.95
Reasoning: The text expresses a strong appreciation for the camera quality, indicating satisfaction and a positive sentiment.

Review: The picture quality is outstanding, best camera ever
Classification: Positive
Confidence: 0.95
Reasoning: The text expresses a strong approval of the picture quality and the camera, indicating a positive sentiment.

Review: Screen keeps freezing during normal use
Classification: Negative
Confidence: 0.85
Reasoning: The text describes a frustrating issue (screen freezing) that negatively impacts the user experience during normal use.

Review: App crashes frequently, very buggy interface
Classification: Negative
Confidence: 0.95
Reasoning: The text expresses dissatisfaction with the app, highlighting issues such as frequent crashes and a buggy interface, which are clear indicators of a negative experience.


 """
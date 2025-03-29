import pandas as pd
import polyflow
from polyflow.models import LanguageProcessor, SentenceVectorizer

def setup_api_credentials():
    """Setup OpenAI API credentials"""
    api_key = "OPEN_AI_API_KEY"
    return api_key

# Configure both language and vector models
lm = LanguageProcessor(model="gpt-4o", api_key=setup_api_credentials())
rm = SentenceVectorizer(model="intfloat/multilingual-e5-large")
polyflow.settings.configure(lm=lm, rm=rm)

# Sample dataset of electric vehicle reviews
reviews_data = {
    "product_id": range(1, 11),
    "review_text": [
        "The range on this EV is disappointing, barely making 200 miles on a full charge despite the 300-mile claim",
        "Battery degradation is significant after just one year, losing almost 15% of original capacity",
        "Charging infrastructure is still inadequate for long trips, spent hours finding working stations",
        "The fast-charging capability is impressive, going from 10% to 80% in just 20 minutes",
        "Regenerative braking system is fantastic, rarely need to use the actual brake pedal",
        "Software updates constantly improve the vehicle, adding new features and fixing issues automatically",
        "Build quality issues are evident - panel gaps, interior rattles, and loose trim pieces",
        "The audio system has terrible sound quality, even after adjusting all settings",
        "Acceleration is phenomenal, effortlessly merging onto highways and passing other vehicles",
        "The autopilot feature works remarkably well on highways, making long drives much less tiring"
    ]
}

df = pd.DataFrame(reviews_data)

# Step 1: Cluster reviews using semantic clustering
clustered_reviews = df.sk_sem_cluster(
    text_column="review_text",
    n_clusters=4,
    method="hierarchical",
    return_scores=True
)

# Step 2: Use language model to generate detailed cluster summaries
def generate_cluster_summary(cluster_reviews):
    prompt = f"""Analyze these electric vehicle reviews to identify key themes, issues, and sentiment:

Reviews:
{chr(10).join(['- ' + review for review in cluster_reviews])}

Please provide:
1. A concise cluster name/label based on the main theme
2. The primary sentiment (positive, negative, or mixed)
3. A detailed analysis of what aspects of electric vehicles these reviews focus on
4. Key issues or highlights mentioned
"""

    messages = [[
        {
            "role": "system",
            "content": "You are an expert at analyzing customer reviews and identifying patterns and insights."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]]
    
    try:
        response = lm(messages)
        if hasattr(response, 'outputs'):
            return response.outputs[0]
        return str(response)
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Group reviews by cluster and generate detailed summaries
print("\nEV Review Cluster Analysis:")
print("=" * 80)

for cluster_id in sorted(clustered_reviews['cluster_id'].unique()):
    cluster_reviews = clustered_reviews[
        clustered_reviews['cluster_id'] == cluster_id
    ]['review_text'].tolist()
    
    print(f"\nCluster {cluster_id} ({len(cluster_reviews)} reviews):")
    print("-" * 60)
    print("Reviews in this cluster:")
    for review in cluster_reviews:
        print(f"• {review}")
    
    print("\nCluster Summary and Analysis:")
    summary = generate_cluster_summary(cluster_reviews)
    print(summary)
    print("-" * 60)

# Step 3: Use semantic aggregation for additional insights
df_with_clusters = clustered_reviews.copy()
insights = df_with_clusters.sem_agg(
    "What are the key EV features and their perceived quality based on these {review_text}?",
    group_by=["cluster_id"]
)

print("\nCluster Feature Analysis:")
print("=" * 80)
print(insights._output)

# Step 4: Prioritize issues using semantic ranking
priority_issues = df_with_clusters.sem_topk(
    "Which {review_text} indicates the most critical issue that EV manufacturers should address first?",
    K=3
)

print("\nTop Priority Issues for EV Manufacturers:")
print("=" * 80)
print(priority_issues[['review_text']])


""" 

Cluster 0:
Reviews:
- The range on this EV is disappointing, barely making 200 miles on a full charge despite the 300-mile claim
- Battery degradation is significant after just one year, losing almost 15% of original capacity
- Charging infrastructure is still inadequate for long trips, spent hours finding working stations
Processing uncached messages: 100%|█████████████████████████ 1/1 LM calls [00:03<00:00,  3.15s/it]

Cluster Summary:
The main issues/themes from the customer reviews are:

1. **Battery Life**: Customers consistently express dissatisfaction with the battery life, noting that it lasts no more than 200 miles on a full charge despite the 300-mile claim.

2. **Charging Infrastructure**: There are complaints about the charging infrastructure, with customers spending hours finding working stations.

Cluster 1:
Reviews:
- The fast-charging capability is impressive, going from 10% to 80% in just 20 minutes
Processing uncached messages: 100%|█████████████████████████ 1/1 LM calls [00:01<00:00,  1.66s/it]

Cluster Summary:
The main theme from the customer reviews is a strong appreciation for the fast-charging capability, with multiple users highlighting that it allows for a quick charge. This suggests that the fast-charging feature is a standout feature of the product, contributing significantly to customer satisfaction.

Cluster 2:
Reviews:
- Regenerative braking system is fantastic, rarely need to use the actual brake pedal
Processing uncached messages: 100%|█████████████████████████ 1/1 LM calls [00:03<00:00,  3.69s/it]

Cluster Summary:
The main issues/themes from the customer reviews regarding the regenerative braking system include:

1. **Performance Reliability**: Customers express satisfaction with the regenerative braking system, noting that they rarely need to use the actual brake pedal.

2. **Energy Efficiency**: The system's ability to regenerate energy during braking is highlighted as a positive feature, indicating that it contributes to the vehicle's energy efficiency.

3. **User Experience**: The regenerative braking system is noted to be fantastic, contributing to a smoother and more comfortable driving experience.

4. **Support and Resolution**: Some users mention difficulties in obtaining effective support or solutions from customer service, highlighting a gap in assistance for resolving the issue.

5. **Impact on Productivity**: The regenerative braking system is noted to be a positive feature, contributing to a smoother and more comfortable driving experience.

Overall, the recurring theme is a strong appreciation for the regenerative braking system, which is a standout feature of the product.

Cluster 3:
Reviews:
- Software updates constantly improve the vehicle, adding new features and fixing issues automatically
- Build quality issues are evident - panel gaps, interior rattles, and loose trim pieces
- The audio system has terrible sound quality, even after adjusting all settings
Processing uncached messages: 100%|█████████████████████████ 1/1 LM calls [00:03<00:00,  3.69s/it]

Cluster Summary:
The main issues/themes from the customer reviews regarding software updates and build quality include:

1. **Software Updates**: Customers express satisfaction with the software updates, noting that they constantly improve the vehicle, adding new features and fixing issues automatically.

2. **Build Quality**: There are complaints about build quality issues, with panel gaps, interior rattles, and loose trim pieces mentioned.

3. **Audio System**: There are complaints about the audio system, with terrible sound quality mentioned.

4. **Support and Resolution**: Some users mention difficulties in obtaining effective support or solutions from customer service, highlighting a gap in assistance for resolving the issue.

5. **Impact on Productivity**: The software updates are noted to be a positive feature, contributing to a smoother and more comfortable driving experience.

Overall, the recurring theme is a significant concern over the vehicle's build quality and the need for improvements in software performance and customer support.
Aggregating: 100%|██████████████████████████████████████████ 1/1 LM calls [00:01<00:00,  1.34s/it]
Aggregating: 100%|██████████████████████████████████████████ 1/1 LM calls [00:01<00:00,  1.91s/it]
Aggregating: 100%|██████████████████████████████████████████ 1/1 LM calls [00:02<00:00,  2.36s/it]
Aggregating: 100%|██████████████████████████████████████████ 1/1 LM calls [00:02<00:00,  2.36s/it]
Detailed Cluster Analysis:
0    The key problems identified in the reviews are...
0    Context: The reviews highlight exceptional cam...
0    The key problem identified is that the screen ...
Name: _output, dtype: object
Quicksort comparisons: 100%|██████████████████████████████████████████| 5/5 LM calls [00:00<00:00]
Quicksort comparisons: 100%|██████████████████████████████████████████| 3/3 LM calls [00:00<00:00]

Top Priority Issues:
                                         review_text
0       App crashes frequently, very buggy interface
1            Screen keeps freezing during normal use
2  The battery life is terrible, doesn't last mor...

 """
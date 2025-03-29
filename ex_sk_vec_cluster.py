import pandas as pd
import polyflow
from polyflow.models import SentenceVectorizer

# Configure the model with a modern multilingual embedding model
rm = SentenceVectorizer(model="intfloat/multilingual-e5-large", normalize_embeddings=True)
polyflow.settings.configure(rm=rm)

# Sample IT incident reports for clustering analysis
incidents_data = {
    "incident_id": range(1, 13),
    "description": [
        "Database server unresponsive after scheduled maintenance window",
        "Production database experiencing high latency and connection timeouts",
        "Multiple users reporting inability to access shared network drives",
        "Network file shares inaccessible from Windows 10 workstations",
        "Corporate website returning 503 Service Unavailable errors",
        "Company website down with connection refused errors",
        "Multiple failed login attempts detected from IP range 192.168.100.x",
        "Suspicious login activity from unauthorized IP addresses",
        "All printers in Building C offline and unreachable",
        "Network printers showing offline status across all departments",
        "Kubernetes cluster autoscaling not functioning properly during traffic spikes",
        "Container orchestration system failing to deploy new application versions"
    ]
}

df = pd.DataFrame(incidents_data)

# Cluster incidents using hierarchical clustering
clustered_incidents = df.sk_sem_cluster(
    text_column="description",
    n_clusters=5,
    method="hierarchical",
    return_scores=True,
    return_centroids=True
)

# Print clusters with similarity scores
print("\nIT Incident Categories by Semantic Similarity:")
print("=" * 80)

# Extract results and cluster details
incident_clusters = clustered_incidents[0]

# Group by cluster and print each category
for cluster_id in range(5):
    cluster_incidents = incident_clusters[incident_clusters['cluster_id'] == cluster_id]
    
    if len(cluster_incidents) > 0:
        # Calculate average similarity within cluster
        avg_similarity = cluster_incidents['centroid_similarity'].mean()
        
        print(f"\nIncident Category {cluster_id} (Avg. Similarity: {avg_similarity:.3f}):")
        print("-" * 60)
        
        for _, incident in cluster_incidents.iterrows():
            print(f"Incident {incident['incident_id']}: {incident['description']}")
            print(f"Similarity Score: {incident['centroid_similarity']:.3f}")
        
        print()

# Calculate metrics about the clustering
cluster_sizes = incident_clusters.groupby('cluster_id').size()
print("\nCluster Size Distribution:")
for cluster_id, size in cluster_sizes.items():
    print(f"Category {cluster_id}: {size} incidents")

# Find most representative incident for each cluster (closest to centroid)
print("\nMost Representative Incidents:")
for cluster_id in range(5):
    if cluster_id in cluster_sizes.index:
        representative = incident_clusters[incident_clusters['cluster_id'] == cluster_id].sort_values(
            'centroid_similarity', ascending=False
        ).iloc[0]
        print(f"Category {cluster_id}: \"{representative['description']}\" (Similarity: {representative['centroid_similarity']:.3f})")



""" 

Issue Category 0:
Ticket 1: Cannot login to my account, password reset not working
Similarity Score: 0.942
Ticket 2: Login page shows error after entering credentials
Similarity Score: 0.942

Issue Category 1:
Ticket 3: App crashes when uploading large files
Similarity Score: 0.956
Ticket 4: Application freezes during file upload
Similarity Score: 0.956

Issue Category 2:
Ticket 5: Billing statement shows incorrect charges
Similarity Score: 0.942
Ticket 6: Wrong amount charged on monthly subscription
Similarity Score: 0.942

 """
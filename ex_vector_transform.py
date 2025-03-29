import pandas as pd
import polyflow
from polyflow.models import SentenceVectorizer

# Initialize and configure modern vector model with better language understanding
rm = SentenceVectorizer(
    model="intfloat/multilingual-e5-large",
    normalize_embeddings=True
)
polyflow.settings.configure(rm=rm)

# Create sample data with diverse tech and non-tech roles
data = {
    "job_titles": [
        "Senior Software Engineer - Cloud Infrastructure",
        "Full Stack Developer - React/Node.js",
        "Machine Learning Engineer - NLP Specialist",
        "Data Scientist - Healthcare Analytics",
        "DevOps Engineer - Kubernetes Expert",
        "Chief Technology Officer",
        "IT Project Manager",
        "UX/UI Designer - Mobile Applications",
        "Product Marketing Manager",
        "Human Resources Director",
        "Content Strategist",
        "Financial Analyst - Investment Banking",
        "Customer Success Manager",
        "Sales Development Representative",
        "Operations Director - Supply Chain"
    ]
}
df = pd.DataFrame(data)

# First create index for the column for efficient searching
df = df.vector_index("job_titles", "job_titles_index")

# Example 1: Find technical programming-focused roles
tech_roles = df.vector_transform(
    "job_titles",
    "technical software development and programming role requiring coding skills",
    K=6,
    threshold=0.80,
    return_scores=True
)
print("\nSoftware Development Roles:")
print("=" * 80)
print(tech_roles[["job_titles", "job_titles_transformed", "job_titles_score"]].sort_values("job_titles_score", ascending=False))

# Example 2: Find management and leadership positions
leadership_roles = df.vector_transform(
    "job_titles",
    "executive leadership, management, or director-level position with people oversight responsibilities",
    threshold=0.78,
    return_scores=True,
    suffix="_leadership"
)
print("\nLeadership & Management Roles:")
print("=" * 80)
print(leadership_roles[["job_titles", "job_titles_leadership", "job_titles_score"]].sort_values("job_titles_score", ascending=False))

# Example 3: Find customer-facing roles
customer_roles = df.vector_transform(
    "job_titles",
    "customer-facing position involving direct customer interaction or supporting customer needs",
    K=4,
    threshold=0.75,
    return_scores=True,
    suffix="_customer"
)
print("\nCustomer-Facing Roles:")
print("=" * 80)
print(customer_roles[["job_titles", "job_titles_customer", "job_titles_score"]].sort_values("job_titles_score", ascending=False))

# Example 4: Find roles matching multiple criteria (technical AND leadership)
technical_leaders = df.vector_transform(
    "job_titles",
    "technical leadership position combining both technical expertise and management responsibilities",
    K=3,
    threshold=0.82,
    return_scores=True,
    suffix="_tech_leader"
)
print("\nTechnical Leadership Roles:")
print("=" * 80)
print(technical_leaders[["job_titles", "job_titles_tech_leader", "job_titles_score"]].sort_values("job_titles_score", ascending=False))



""" 

Output:

Tech-focused roles:
           job_titles       job_titles_transformed  job_titles_score
0   Software Engineer            Software Engineer              0.88
1      Data Scientist               Data Scientist              0.80
2         ML Engineer                  ML Engineer              0.82
3  Frontend Developer                      <NA>                 <NA>
4     DevOps Engineer              DevOps Engineer              0.76
5    Product Manager                      <NA>                 <NA>
6        UX Designer                      <NA>                 <NA>
7  Backend Developer            Backend Developer              0.78

Management roles:
           job_titles    job_titles_management
0   Software Engineer                      <NA>
1      Data Scientist                      <NA>
2         ML Engineer                      <NA>
3  Frontend Developer                      <NA>
4     DevOps Engineer                      <NA>
5    Product Manager           Product Manager
6        UX Designer                      <NA>
7  Backend Developer                      <NA>

Design roles:
           job_titles       job_titles_transformed
0   Software Engineer                       <NA>
1      Data Scientist                       <NA>
2         ML Engineer                       <NA>
3  Frontend Developer                       <NA>
4     DevOps Engineer                       <NA>
5    Product Manager                        <NA>
6        UX Designer               UX Designer
7  Backend Developer                        <NA>

 """


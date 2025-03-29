import pandas as pd
import polyflow
from polyflow.models import LanguageProcessor
import os

def setup_api_credentials():
    """Setup OpenAI API credentials"""
    api_key = "OPENAI_API_KEY"
    return api_key

def setup_llm():
    """Setup language processor with custom settings"""
    try:
        language_processor = LanguageProcessor(
            model="gpt-4o",
            api_key=setup_api_credentials(),
            temperature=0.1,
            max_tokens=2048,
        )
        return language_processor
    except Exception as e:
        raise Exception(f"Failed to initialize language processor: {str(e)}")

def main():
    # Initialize and configure language processor
    language_processor = setup_llm()
    polyflow.settings.configure(lm=language_processor)

    # Example 1: Academic Paper Classification
    papers_data = {
        "paper_title": [
            "Deep Reinforcement Learning for Autonomous Vehicle Control",
            "The Ethics of Artificial Intelligence in Healthcare Settings",
            "Quantum Computing: A Survey of Recent Advances",
            "Climate Change Impact on Ocean Biodiversity",
            "Privacy Concerns in Facial Recognition Technology",
            "Algorithms for Natural Language Understanding",
            "New Approaches to CRISPR Gene Editing",
            "The Cultural Impact of Social Media on Generation Z",
            "Advancements in Solar Panel Efficiency",
            "Machine Learning in Financial Fraud Detection"
        ]
    }
    papers_df = pd.DataFrame(papers_data)

    # Classify papers as technical vs. non-technical
    technical_papers = papers_df.llm_transform(
        "paper_title",
        "Is this academic paper primarily technical/engineering-focused rather than theoretical/conceptual?",
        threshold=0.75,
        return_scores=True,
        return_reasoning=True,
        suffix="_technical"
    )
    print("\nTechnical Paper Classification:")
    print(technical_papers[["paper_title", "paper_title_technical", "paper_title_score", "paper_title_reasoning"]].sort_values("paper_title_score", ascending=False))

    # Example 2: Job Posting Analysis
    job_data = {
        "job_description": [
            "Seeking an experienced data scientist with expertise in machine learning and statistical analysis. Must have strong Python skills and experience with deep learning frameworks.",
            "Marketing manager needed for growing e-commerce brand. Responsibilities include social media strategy, campaign management, and tracking ROI on marketing initiatives.",
            "Software engineer with 3+ years of experience in full-stack development. Knowledge of React, Node.js, and AWS required.",
            "Healthcare administrator to oversee daily operations of a medical practice. Experience with electronic health records and insurance billing preferred.",
            "Research assistant position in climate science laboratory. Will assist with data collection, sample analysis, and literature reviews.",
            "UX/UI designer to create intuitive digital experiences. Must be proficient in Figma and have a strong portfolio of previous work.",
            "Financial analyst needed for investment firm. Responsibilities include market research, financial modeling, and portfolio analysis."
        ]
    }
    jobs_df = pd.DataFrame(job_data)

    # Identify technical skills required in each job
    tech_skills = jobs_df.llm_transform(
        "job_description",
        "What are the primary technical skills required for this job? Return a comma-separated list",
        return_reasoning=True,
        suffix="_skills"
    )
    print("\nTechnical Skills Analysis:")
    for _, row in tech_skills.iterrows():
        print(f"\nJob: {row['job_description'][:100]}...")
        print(f"Skills: {row['job_description_skills']}")
        print(f"Reasoning: {row['job_description_reasoning']}")

    # Print total language model usage
    print("\nTotal Language Model Usage:")
    language_processor.print_total_usage()

if __name__ == "__main__":
    main()
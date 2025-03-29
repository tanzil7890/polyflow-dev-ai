import pandas as pd
import polyflow
from polyflow.models import LanguageProcessor
import os

def setup_api_credentials():
    """Setup OpenAI API credentials"""
    api_key = "OPEN_AI_API_KEY"
    
    # Set environment variables
    #os.environ["OPENAI_API_KEY"] = api_key
    
    return api_key

def main():
    # Setup API credentials
    api_key = setup_api_credentials()
    
    # Configure language processor with credentials
    language_processor = LanguageProcessor(
        model="gpt-4o",
        api_key=api_key,
        temperature=0.2
    )
    polyflow.settings.configure(lm=language_processor)

    # Create dataframes with university courses and career skills
    courses_data = {
        "Course Name": [
            "Introduction to Artificial Intelligence",
            "Advanced Calculus and Mathematical Analysis",
            "Database Systems Architecture",
            "Sustainable Urban Development",
            "Principles of Biochemistry",
            "Social Media Marketing Strategies",
            "Cognitive Psychology and Neuroscience",
            "Quantum Computing Fundamentals",
            "International Business Law",
            "Creative Writing Workshop"
        ]
    }
    skills_data = {
        "Skill": [
            "Data Analysis", 
            "Mathematical Reasoning", 
            "Software Development",
            "Project Management",
            "Research Methodology",
            "Communication",
            "Critical Thinking"
        ]
    }
    courses_df = pd.DataFrame(courses_data)
    skills_df = pd.DataFrame(skills_data)

    # PolyFlow semantic join with a more nuanced prompt
    print("\nPerforming semantic join analysis...")
    res = courses_df.vector_join(
        skills_df, 
        "Analyze to what extent taking the course '{Course Name}' would help develop the skill '{Skill}'. "
        "Consider both direct and indirect relationships between course content and skill development."
    )
    
    # Filter and display highest relevance matches
    print("\nTop Course-Skill Relationships:")
    if 'score' in res.columns:
        sorted_res = res.sort_values('score', ascending=False).head(10)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        print(sorted_res[['Course Name', 'Skill', 'score']])
    else:
        print(res)

    # Print total LM usage
    print("\nTotal LM Usage:")
    language_processor.print_total_usage()

if __name__ == "__main__":
    main()
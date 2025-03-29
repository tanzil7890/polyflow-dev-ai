import pandas as pd
import polyflow
from polyflow.models import LlamaProcessor
import os

def setup_llama():
    """Setup Llama model using Ollama"""
    try:
        # Configure Llama processor to use Ollama's API with specific system prompt
        llama_processor = LlamaProcessor(
            model_path="ollama/llama3",
            generation_temperature=0.1,
            response_max_tokens=2048,
            system_prompt=(
                "You are an expert at analyzing relationships between academic courses and professional skills. "
                "For each course-skill pair, analyze if studying the course directly contributes to developing the skill. "
                "Return a JSON with a 'score' between 0 and 1, where:\n"
                "1.0: Essential and direct relationship (e.g., 'Machine Learning' course and 'AI Development' skill)\n"
                "0.8-0.9: Strong and clear relationship (e.g., 'Database Design' and 'SQL Programming')\n"
                "0.5-0.7: Moderate or partial relationship\n"
                "0.1-0.4: Weak or tangential relationship\n"
                "0.0: No meaningful relationship\n"
                "Only return scores above 0.7 for courses with direct skill-building potential."
            )
        )
        return llama_processor
    except Exception as e:
        raise Exception(f"Failed to initialize Llama model: {str(e)}")

def main():
    try:
        print("Setting up Llama processor...")
        llama_processor = setup_llama()
        polyflow.settings.configure(lm=llama_processor)
        print("Model loaded successfully!")

        # Create dataframes with modern STEM courses and technical skills
        courses_data = {
            "Course Name": [
                "Machine Learning Fundamentals",
                "Advanced Algorithms and Data Structures",
                "Web Application Development",
                "Cloud Computing Architecture",
                "Database Systems Design",
                "Computer Vision and Deep Learning",
                "DevOps and Continuous Integration",
                "Quantum Computing Principles",
                "Cybersecurity Risk Management",
                "Blockchain Development"
            ]
        }
        
        skills_data = {
            "Skill": [
                "Python Programming",
                "Data Analysis",
                "Full-Stack Development",
                "Cloud Infrastructure Management",
                "Database Administration",
                "AI/ML Implementation",
                "Network Security"
            ]
        }
        
        courses_df = pd.DataFrame(courses_data)
        skills_df = pd.DataFrame(skills_data)

        # PolyFlow semantic join with enhanced prompt for Llama model
        print("\nPerforming semantic join analysis...")
        res = courses_df.vector_join(
            skills_df, 
            "Analyze how directly studying {Course Name} develops competency in {Skill}. "
            "Consider only concrete, practical skill development, not theoretical knowledge. "
            "For example:\n"
            "- 'Database Systems Design' and 'Database Administration' should have a high score\n"
            "- 'Machine Learning Fundamentals' and 'AI/ML Implementation' should have a high score\n"
            "- 'Blockchain Development' and 'Network Security' might have a moderate score\n"
            "Return a JSON with a 'score' field indicating the strength of the relationship."
        )
        
        # Filter and format results
        print("\nAnalysis Results:")
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        
        # Post-process to filter meaningful relationships (score > 0.7)
        if 'score' in res.columns:
            filtered_res = res[res['score'] > 0.7].sort_values('score', ascending=False)
            if not filtered_res.empty:
                print("Strong Course-Skill Relationships:")
                print(filtered_res[['Course Name', 'Skill', 'score']])
            else:
                print("No strong relationships found")
                
            # Show some moderate relationships too
            moderate_res = res[(res['score'] > 0.4) & (res['score'] <= 0.7)].sort_values('score', ascending=False).head(5)
            if not moderate_res.empty:
                print("\nModerate Course-Skill Relationships:")
                print(moderate_res[['Course Name', 'Skill', 'score']])
        else:
            print("Error: No score column found in results")
            print("Raw results:")
            print(res)

        print("\nTotal LM Usage Statistics:")
        llama_processor.print_total_usage()

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nDebug information:")
        if 'res' in locals():
            print("\nRaw response structure:")
            print(res.columns if hasattr(res, 'columns') else "No columns attribute")
            print("\nFirst few rows:")
            print(res.head() if hasattr(res, 'head') else "No head method")

if __name__ == "__main__":
    main()
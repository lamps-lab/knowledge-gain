import json
import os
from openai import OpenAI

# Initialize the OpenAI client. 
client = OpenAI()

def drafter_agent(abstract: str, news_length: int) -> str:
    """Agent 1: Translates the abstract into an accessible news draft."""
    system_prompt = (
        "You are an expert science journalist. Your job is to take complex academic abstracts "
        "and turn them into engaging, accessible news articles for the general public. "
        "Create a catchy headline, keep the tone informative and exciting, and make the "
        "science easy to understand without dumbing it down."
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please draft a news article in {news_length} words based on this abstract:\n\n{abstract}"}
        ],
        temperature=0.7 # Higher temperature for creative writing
    )
    return response.choices[0].message.content

def revision_agent(abstract: str, draft: str) -> str:
    """Agent 2: Reviews the draft against the abstract and improves it."""
    system_prompt = (
        "You are a strict but fair senior editor at a top science magazine. "
        "Your job is to review article drafts to ensure they are factually accurate based "
        "on the original abstract, check for readability, and improve the narrative flow. "
        "Do not introduce new facts, statistics, or claims outside of the abstract."
    )
    
    user_prompt = f"""
    ORIGINAL ABSTRACT:
    {abstract}
    
    INITIAL DRAFT:
    {draft}
    
    Please revise and polish the draft. Ensure no scientific inaccuracies were introduced, 
    improve the hook, and output ONLY the final polished article (including the headline).
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2 # Lower temperature for a more grounded, analytical edit
    )
    return response.choices[0].message.content

def generate_news_workflow(abstract: str, news_length: int) -> str:
    """Main function orchestrating the agentic workflow."""
    final_article = ""
    for i in range(2):
        print(f"iteration {i+1}")
        print("  ✍️  Drafter Agent is analyzing the abstract and writing the initial draft...")
        initial_draft = drafter_agent(abstract, news_length)
    
        print("  🧐  Revision Agent is fact-checking and polishing the draft...")
        final_article = revision_agent(abstract, initial_draft)
    
    return final_article

def main():
    # Path to JSON dataset
    file_path = "twenty_samples.json"
    
    # Load the JSON data
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{file_path}'. Please check the path.")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: '{file_path}' is not a valid JSON file.")
        return
    
    #print(f"{len(dataset)} pairs")
    #exit()
    
    # Process each entry in the dataset
    for i, entry in enumerate(dataset):
        article_id = entry.get('id', 'Unknown ID')
        category = entry.get('category', 'Unknown Category')
        abstract = entry.get('abstract', '')
        news = entry.get('news', '')
        news_length = len(news.split())
        
        print(f"\n{'='*60}")
        print(f"Processing Entry {i+1}/{len(dataset)} | ID: {article_id} | Category: {category} | {news_length} words")
        print(f"{'='*60}")
        
        # Skip if the abstract is empty or if it contains our scraper error messages
        if not abstract or "Manual extraction needed" in abstract:
            print("  ⚠️  Skipping: No valid abstract text found for this entry.")
            continue
            
        # Run the multi-agent workflow
        final_news = generate_news_workflow(abstract, news_length)
        
        print("\n--- FINAL PUBLISHED ARTICLE ---\n")
        print(final_news)
        print("\n" + "-"*60)

        entry['generated_news'] = final_news

    with open("evaluation/agentic_news.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
    print("\n✅ All finished! Saved results to agentic_news.json")

if __name__ == "__main__":
    main()
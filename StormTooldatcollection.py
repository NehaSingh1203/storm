import os
from knowledge_storm import STORMWikiRunnerArguments, STORMWikiLMConfigs

from knowledge_storm import STORMWikiRunner

from knowledge_storm.lm import OpenAIModel
from knowledge_storm import GoogleSearch
import csv

class CustomSTORMWikiRunner(STORMWikiRunner):
    def __init__(self, engine_args, lm_configs, search_runner):
        super().__init__(engine_args, lm_configs, search_runner)

    def run_multiple_terms(self, terms, standard_prompt):
        results = []

        for term in terms:
            topic = f"{standard_prompt} {term}"
            try:
                self.run(
                    topic=topic,
                    do_research=True,
                    do_generate_outline=True,
                    do_generate_article=True,
                    do_polish_article=True,
                )
                self.post_run()
                article_content = self.summary()
                results.append((term, article_content))
            except Exception as e:
                print(f"Error processing term '{term}': {e}")
                results.append((term, "Error occurred during processing."))

        return results

    def save_to_csv(self, results, output_path):
        try:
            with open(output_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Search Term', 'Article Content'])
                for term, content in results:
                    content = self.replace_sources(content)
                    writer.writerow([term, content])
        except Exception as e:
           print(f"Failed to save results to CSV: {e}")

    def replace_sources(self, content):
        # Replace citations with custom logic
        return content.replace("(1)", "Source A").replace("(2)", "Source B")
 

os.environ['OPENAI_API_KEY'] = 'your_openai_key'
os.environ['GOOGLE_SEARCH_API_KEY'] = "API_key"
os.environ['GOOGLE_CSE_ID'] = "CSE_ID"

# Configuration for OpenAI models
lm_configs = STORMWikiLMConfigs()
openai_kwargs = {
    'api_key': os.getenv("OPENAI_API_KEY"),
    'temperature': 1.0,
    'top_p': 0.9,
}

# Cheaper/faster model for conversation and question splitting
gpt_35 = OpenAIModel(model='gpt-3.5-turbo', max_tokens=500, **openai_kwargs)
# More powerful model for generating articles
gpt_4 = OpenAIModel(model='gpt-4o', max_tokens=3000, **openai_kwargs)

# Assigning models to different components
lm_configs.set_conv_simulator_lm(gpt_35)
lm_configs.set_question_asker_lm(gpt_35)
lm_configs.set_outline_gen_lm(gpt_4)
lm_configs.set_article_gen_lm(gpt_4)
lm_configs.set_article_polish_lm(gpt_4)
engine_args = STORMWikiRunnerArguments(output_dir="F:\SearchStrategyProject\Stormtool")
# Instantiate the search runner with Google Search API
rm = GoogleSearch(google_search_api_key=os.getenv("GOOGLE_SEARCH_API_KEY"), google_cse_id=os.getenv("GOOGLE_CSE_ID"),user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3')
def main():
    # Configuration and initialization...
    
    search_terms = read_search_terms("F:\SearchStrategyProject\Stormtool/search_terms.txt")  # Implement this function
    standard_prompt = read_standard_prompt("F:\SearchStrategyProject\Stormtool/standard_prompt.txt")  # Implement this function

    runner = CustomSTORMWikiRunner(engine_args, lm_configs, rm)
    results = runner.run_multiple_terms(search_terms, standard_prompt)
    
    # Save results to CSV
    runner.save_to_csv(results, "F:\SearchStrategyProject\Stormtool/articles.csv")

def read_search_terms(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def read_standard_prompt(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()

if __name__ == "__main__":
    main()

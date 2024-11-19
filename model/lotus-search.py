import json
import arxiv
import pandas as pd
import numpy as np
import gradio as gr
import lotus
import json, re, time, os
from datetime import datetime, timedelta
from lotus.models import LM



OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    raise Exception("OpenAI API Key is required")

# Configure Lotus
lm = LM(model="gpt-4o-mini")
lotus.settings.configure(
    lm=lm,
    rm=lotus.models.LiteLLMRM(model="text-embedding-3-small")
)

def index_columns(dataframe):
    print("Creating similarity indices for columns...")
    columns_to_index = ["projects", "skills", "awards", "education", "workExperience", "certifications", "personalInformation"]
    for column in columns_to_index:
        dataframe = dataframe.sem_index(column, f"model/index_{column}_dir")
    print("Indices created successfully!")
    return dataframe

def load_dataframe():
    file_path = os.path.join('data', 'all_resumes.json')
    with open(file_path, 'r') as file:
        data = json.load(file)
        df = pd.json_normalize(data, sep='_')
        sampled_df = df.head(10)
    return sampled_df

def flatten_to_string(value):
    if isinstance(value, (list, dict)):
        return str(value)
    return value 

def query_across_columns(dataframe, query, columns):
    print(f"Searching across columns: {', '.join(columns)} for query: {query}")
    results = []
    
    for column in columns:
        dataframe = dataframe.load_sem_index(column, f"model/index_{column}_dir")
        column_results = dataframe.sem_search(column, query, 10)        
        column_results = column_results.map(flatten_to_string)
        
        results.append(column_results)
    
    combined_results = pd.concat(results, ignore_index=True)
    combined_results = combined_results.map(flatten_to_string).drop_duplicates().reset_index(drop=True)
    
    return combined_results

def preprocess_data(dataframe):
    text_columns = ["projects", "skills", "awards", "education", "workExperience", "certifications"]
    for column in text_columns:
        dataframe[column] = dataframe[column].fillna('').astype(str)
    
    dataframe = dataframe[
        (dataframe["skills"] != "") |
        (dataframe["workExperience"] != "") |  
        (dataframe["awards"] != "") | 
        (dataframe["certifications"] != "") | 
        (dataframe["projects"] != "")
    ]
    
    return dataframe

def gradio_query(query_type, query_value):
    dataframe = load_dataframe()
    dataframe = preprocess_data(dataframe)

    if query_type == "Skill-based Search":
        results = query_across_columns(dataframe, query_value, ["projects", "skills", "certifications", "workExperience", "awards"])
    elif query_type == "Education-based Search":
        results = query_across_columns(dataframe, query_value, ["education", "certifications", "personalInformation"])
    elif query_type == "Role-based Search":
        results = query_across_columns(dataframe, query_value, ["projects", "workExperience", "certifications"])
    else:
        return "Invalid choice!"

    if results.empty:
        return "No matching results found."
    else:
        results_file_path = os.path.join('output', 'results.csv')
        return results.to_csv(results_file_path, index=False)

def gradio_query(query_type, query_value):
    dataframe = load_dataframe()
    dataframe = preprocess_data(dataframe)
    
    column_mappings = {
        "Skill-based Search": ["projects", "skills", "certifications", "workExperience", "awards"],
        "Education-based Search": ["education", "certifications", "personalInformation"],
        "Role-based Search": ["projects", "workExperience", "certifications"]
    }
    
    if query_type not in column_mappings:
        return None, "Invalid search type selected. Please try again."
        
    if not query_value.strip():
        return None, "Please enter a search query."
        
    results = query_across_columns(dataframe, query_value, column_mappings[query_type])
    
    if results.empty:
        return None, "No matching results found."
        
    results_file_path = os.path.join('output', 'results.csv')
    results.to_csv(results_file_path, index=False)
    return results_file_path, f"Found {len(results)} matching results."

def create_gradio_interface():
    with gr.Blocks(theme=gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="gray",
        font=["Inter", "ui-sans-serif", "system-ui"]
    )) as demo:
        gr.Markdown("""
        # Resume Search Tool
        Search through resumes based on skills, education, or roles.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                query_type = gr.Dropdown(
                    choices=[
                        "Skill-based Search",
                        "Education-based Search",
                        "Role-based Search"
                    ],
                    label="Search Type",
                    value="Skill-based Search"
                )
                
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter your search terms...",
                    lines=2
                )
                
                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    submit_btn = gr.Button("Search", variant="primary")
            
            with gr.Column(scale=3):
                output_file = gr.File(label="Results")
                status_text = gr.Textbox(label="Status", interactive=False)
        
        # Button actions
        submit_btn.click(
            fn=gradio_query,
            inputs=[query_type, query_input],
            outputs=[output_file, status_text]
        )
        
        clear_btn.click(
            fn=lambda: (None, None, ""),
            inputs=[],
            outputs=[output_file, status_text, query_input]
        )
        
        # Example queries
        gr.Examples(
            examples=[
                ["Skill-based Search", "python machine learning"],
                ["Education-based Search", "computer science"],
                ["Role-based Search", "software engineer"]
            ],
            inputs=[query_type, query_input]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
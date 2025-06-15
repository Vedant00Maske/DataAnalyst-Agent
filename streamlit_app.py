import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import together
import time
import os
import io
import pytesseract
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
from dotenv import load_dotenv

# Load API key
load_dotenv()
together.api_key = os.getenv("TOGETHER_API_KEY")

st.set_page_config(page_title="Data Analyst Agent", layout="wide")
st.title("Data Analyst Agent")

# Cache for performance
@st.cache_data
def query_llama(prompt):
    try:
        client = together.Client()
        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {"role": "system", "content": "You are a professional data analyst providing concise, direct answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=512,
            top_p=0.7,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        if "429" in str(e):
            st.warning("Rate limit hit. Waiting for 60 seconds...")
            time.sleep(60)
            return query_llama(prompt)
        return f"[ERROR]: {str(e)}"

# File reading functions
def read_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

def read_excel(uploaded_file):
    return pd.read_excel(uploaded_file)

def read_txt(uploaded_file):
    content = uploaded_file.getvalue().decode("utf-8")
    return content

def read_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(io.BytesIO(uploaded_file.getvalue())) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def read_image(uploaded_file):
    img = Image.open(io.BytesIO(uploaded_file.getvalue()))
    return pytesseract.image_to_string(img)

# Data Summary & QnA Functions
def get_sample_summary(df):
    return f"Columns: {df.columns.tolist()}\nHead:\n{df.head().to_string()}"

def ask_question_about_data(df, user_question):
    """
    Get concise, professional answers to questions about the data
    """
    # Include more dataset statistics to give the model better context
    stats_summary = ""
    try:
        stats_summary = f"""
Basic statistics for numeric columns:
{df.describe().to_string()}

Dataset shape: {df.shape}
"""
    except:
        pass
        
    # Enhanced prompt with instructions for direct, professional answers
    summary = get_sample_summary(df)
    prompt = f"""You are a professional data analyst responding to executive-level questions. 
Given the following dataset information:

{summary}

{stats_summary}

Respond to this question with a DIRECT and CONCISE answer:
Q: {user_question}

IMPORTANT INSTRUCTIONS:
1. Provide ONLY the final answer without showing calculation steps
2. Be precise and quantitative where appropriate
3. Format numbers professionally (with thousands separators, appropriate decimal places)
4. DO NOT explain your process or methodology 
5. Keep your answer to 1-3 sentences maximum
6. If creating a list, use bullet points
7. If specific data is requested, provide exact values from the dataset
"""
    return query_llama(prompt)

def generate_visualization(df, request):
    """
    Generate visualizations based on user requests.
    """
    # Get visualization type and parameters from LLM
    prompt = f"""You are a data visualization expert. Parse the following visualization request and extract exactly:
    1. Plot type (histogram, bar chart, scatter plot, line plot, box plot, heatmap, pie chart)
    2. Column(s) to use - MUST match exactly from the available columns
    3. Any other relevant parameters

    Dataset columns: {df.columns.tolist()}
    Data types: {df.dtypes.to_dict()}
    
    User request: "{request}"
    
    Return ONLY a valid JSON structure without any explanation:
    {{
        "plot_type": "type of plot",
        "columns": ["exact_column_name1", "exact_column_name2"],
        "parameters": {{"param1": "value1"}}
    }}
    
    The columns MUST be from the available columns list and spelled exactly the same.
    Do not include parameters unless explicitly mentioned in the request.
    """
    
    with st.spinner("Analyzing your visualization request..."):
        response = query_llama(prompt)
    
    try:
        # Extract JSON-like part from the response
        import re
        import json
        
        # Try to find a JSON pattern
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            
            # Try to parse the JSON
            try:
                viz_params = json.loads(json_str)
            except json.JSONDecodeError as je:
                # If there's an error, try to clean up the JSON string
                cleaned_json = json_str.replace("'", '"')
                viz_params = json.loads(cleaned_json)
        else:
            return None, f"Couldn't parse visualization parameters from response. Please try a different request."
            
        plot_type = viz_params.get("plot_type", "").lower()
        columns = viz_params.get("columns", [])
        parameters = viz_params.get("parameters", {})
        
        # Validate columns
        for col in columns:
            if col not in df.columns:
                return None, f"Column '{col}' not found in the dataset. Available columns: {df.columns.tolist()}"
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create appropriate visualization
        if plot_type == "histogram":
            if len(columns) == 0:
                return None, "No columns specified for histogram"
            sns.histplot(data=df, x=columns[0], ax=ax, **parameters)
            plt.title(f"Histogram of {columns[0]}")
            
        elif plot_type == "bar chart":
            if len(columns) < 1:
                return None, "Need at least one column for bar chart"
            x = columns[0]
            y = columns[1] if len(columns) > 1 else None
            
            if y:
                sns.barplot(data=df, x=x, y=y, ax=ax, **parameters)
                plt.title(f"Bar Chart of {y} by {x}")
            else:
                sns.countplot(data=df, x=x, ax=ax, **parameters)
                plt.title(f"Count of {x}")
                
        elif plot_type == "scatter plot":
            if len(columns) < 2:
                return None, "Need at least two columns for scatter plot"
            x, y = columns[0], columns[1]
            hue = columns[2] if len(columns) > 2 else None
            
            sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, **parameters)
            plt.title(f"Scatter Plot of {y} vs {x}")
            
        elif plot_type == "line plot":
            if len(columns) < 2:
                return None, "Need at least two columns for line plot"
            x, y = columns[0], columns[1]
            
            sns.lineplot(data=df, x=x, y=y, ax=ax, **parameters)
            plt.title(f"Line Plot of {y} vs {x}")
            
        elif plot_type == "box plot":
            if len(columns) < 1:
                return None, "Need at least one column for box plot"
            x = columns[0] if len(columns) > 0 else None
            y = columns[1] if len(columns) > 1 else None
            
            sns.boxplot(data=df, x=x, y=y, ax=ax, **parameters)
            if x and y:
                plt.title(f"Box Plot of {y} by {x}")
            else:
                plt.title(f"Box Plot of {x or y}")
                
        elif plot_type == "heatmap":
            if "correlation" in request.lower():
                corr = df.select_dtypes(include=['number']).corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, **parameters)
                plt.title("Correlation Heatmap")
            else:
                if len(columns) < 2:
                    return None, "Need at least two columns for heatmap"
                pivot_data = df.pivot_table(index=columns[0], columns=columns[1], 
                                           values=columns[2] if len(columns) > 2 else None, 
                                           aggfunc='mean')
                sns.heatmap(pivot_data, annot=True, cmap='coolwarm', ax=ax, **parameters)
                plt.title(f"Heatmap of {columns}")
                
        elif plot_type == "pie chart":
            if len(columns) < 1:
                return None, "Need at least one column for pie chart"
            counts = df[columns[0]].value_counts()
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%', **parameters)
            plt.title(f"Pie Chart of {columns[0]}")
            
        else:
            return None, f"Unsupported plot type: {plot_type}"
        
        plt.tight_layout()
        return fig, f"Created {plot_type} for columns: {', '.join(columns)}"
        
    except Exception as e:
        import traceback
        return None, f"Error generating visualization: {str(e)}"

def handle_data_request(df, request):
    """
    Process user request about data - either answer questions or create visualizations
    """
    # Determine if this is a visualization request or a question
    prompt = f"""Determine if the following request is asking for:
    1. A visualization/chart/graph/plot
    2. A data question/query/analysis
    
    Request: "{request}"
    
    Answer with just "visualization" or "question":"""
    
    with st.spinner("Analyzing your request..."):
        request_type = query_llama(prompt).strip().lower()
    
    if "visual" in request_type or "chart" in request_type or "graph" in request_type or "plot" in request_type:
        return "visualization", request
    else:
        return "question", request

# Sidebar for file upload and basic settings
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt", "pdf", "jpg", "png"])
    
    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1]
        st.success(f"Uploaded {uploaded_file.name}")

# Main area
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_type' not in st.session_state:
    st.session_state.data_type = None

# Process uploaded file
if uploaded_file is not None and (st.session_state.uploaded_file != uploaded_file.name):
    st.session_state.uploaded_file = uploaded_file.name
    
    try:
        with st.spinner("Processing your file..."):
            if file_type == "csv":
                st.session_state.data = read_csv(uploaded_file)
                st.session_state.data_type = "tabular"
            elif file_type == "xlsx":
                st.session_state.data = read_excel(uploaded_file)
                st.session_state.data_type = "tabular"
            elif file_type == "txt":
                st.session_state.data = read_txt(uploaded_file)
                st.session_state.data_type = "text"
            elif file_type == "pdf":
                st.session_state.data = read_pdf(uploaded_file)
                st.session_state.data_type = "text"
            elif file_type in ["jpg", "png"]:
                st.session_state.data = read_image(uploaded_file)
                st.session_state.data_type = "text"
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.session_state.data = None

# Display data and interaction interface
if st.session_state.data is not None:
    if st.session_state.data_type == "tabular":
        df = st.session_state.data
        
        # Display basic data info
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Rows:** {df.shape[0]}")
        with col2:
            st.write(f"**Columns:** {df.shape[1]}")
            
        # Data request interface
        st.subheader("Ask Questions or Request Visualizations")
        user_request = st.text_input("Enter your question or visualization request:", 
                                     placeholder="E.g., 'What is the average price?' or 'Create a histogram of price'")
        
        if user_request:
            request_type, final_request = handle_data_request(df, user_request)
            
            if request_type == "question":
                with st.spinner("Finding your answer..."):
                    answer = ask_question_about_data(df, final_request)
                st.subheader("Answer")
                st.write(answer)
                
            elif request_type == "visualization":
                with st.spinner("Creating visualization..."):
                    fig, result_msg = generate_visualization(df, final_request)
                
                if fig:
                    st.subheader("Visualization")
                    st.pyplot(fig)
                    st.caption(result_msg)
                else:
                    st.error(result_msg)
        
        # Quick action buttons
        st.subheader("Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Data Summary"):
                with st.spinner("Generating summary..."):
                    summary = ask_question_about_data(df, "Provide a brief summary of this dataset")
                st.info(summary)
                
        with col2:
            if st.button("Correlation Heatmap"):
                with st.spinner("Creating correlation heatmap..."):
                    fig, result_msg = generate_visualization(df, "Create a correlation heatmap")
                if fig:
                    st.pyplot(fig)
                else:
                    st.error(result_msg)
                    
        with col3:
            if st.button("Column Distribution"):
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    selected_col = st.selectbox("Select column", numeric_cols)
                    fig, result_msg = generate_visualization(df, f"Create a histogram of {selected_col}")
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.error(result_msg)
                else:
                    st.error("No numeric columns found")
        
    else:  # Text data
        text_data = st.session_state.data
        
        # Display text preview
        st.subheader("Text Preview")
        st.text_area("Content preview", text_data[:500] + "...", height=200)
        
        # Text question interface
        st.subheader("Ask Questions About the Text")
        text_question = st.text_input("Enter your question about the text:", placeholder="E.g., 'What is this document about?'")
        
        if text_question:
            with st.spinner("Finding your answer..."):
                prompt = f"You are an assistant. Here is a document:\n{text_data[:3000]}...\nAnswer this question: {text_question}"
                answer = query_llama(prompt)
            st.subheader("Answer")
            st.write(answer)
            
        # Quick text analysis actions
        st.subheader("Quick Text Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Summarize Document"):
                with st.spinner("Generating summary..."):
                    prompt = f"You are an assistant. Summarize this document in 3-5 sentences:\n{text_data[:3000]}..."
                    summary = query_llama(prompt)
                st.info(summary)
                
        with col2:
            if st.button("Extract Key Points"):
                with st.spinner("Extracting key points..."):
                    prompt = f"You are an assistant. Extract 5 key points from this document:\n{text_data[:3000]}..."
                    key_points = query_llama(prompt)
                st.info(key_points)

else:
    st.info("👈 Please upload a file to get started.")

# Add helpful information in the sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("About This App")
    st.write("This Data Analyst Agent can:")
    st.write("- Import various file types (CSV, Excel, PDF, images)")
    st.write("- Answer questions about your data")
    st.write("- Create visualizations based on natural language requests")
    st.write("- Provide insights and summaries")
    
    st.markdown("---")
    st.write("Example Questions:")
    st.info("What's the average price in the dataset?")
    st.info("How many records are in each category?")
    st.write("Example Visualization Requests:")
    st.info("Create a histogram of price")
    st.info("Show a scatter plot of area vs price")
    st.info("Make a bar chart of property types")
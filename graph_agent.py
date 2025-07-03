# graph_agent.py
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = init_chat_model("llama3-8b-8192", model_provider="groq")


def generate_insight_report(df_head: str, df_dtypes: str) -> str:
    """Generate natural language insights from the dataframe"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data analyst who explains data in clear, concise language."),
        ("human", f"""
You are analyzing the following pandas DataFrame (`df`) preview and column types.

Data preview:
{df_head}

Column types:
{df_dtypes}

Provide a concise insights report (5–10 bullet points) summarizing:
- Data structure
- Key patterns
- Notable values, correlations, or outliers
- Anything interesting worth highlighting

Only return the plain insight bullets. Do not include code or any file loading instructions.
""")
    ])
    response = llm.invoke(prompt.format_messages())
    return response.content.strip()


def generate_insights_and_visualizations_code(df_head: str, df_dtypes: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior Python data analyst and visualization expert."),
        ("human", f"""
You are given a pandas DataFrame called `df`. Your task is to:
1. Analyze the dataset and extract useful insights.
2. Create 4–6 meaningful Plotly visualizations (bar, line, histogram, boxplot, scatter, pie, etc.).

Here is a sample of the dataset:
{df_head}

Here are the column data types:
{df_dtypes}

Generate valid Python code with the following constraints:
- Use Plotly (either `plotly.express` or `plotly.graph_objects`) only.
- Assign each chart to a variable named `fig1`, `fig2`, ..., up to `fig6`.
- Do NOT load or read any CSV or file (e.g., do not use `pd.read_csv`, `open()`, etc.).
- Do NOT redefine or reassign the DataFrame.
- Do NOT print anything. No markdown, comments, or explanations — just Python code inside a single ```python block.
""")
    ])
    response = llm.invoke(prompt.format_messages())
    code = response.content.strip()

    if code.startswith("```"):
        code = code.strip("```").strip()
        if code.startswith("python"):
            code = code[len("python"):].strip()
    return code

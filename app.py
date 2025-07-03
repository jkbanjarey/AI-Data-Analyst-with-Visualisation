# app.py
import os
import streamlit as st
import pandas as pd
import traceback
from dotenv import load_dotenv
from graph_agent import generate_insight_report, generate_insights_and_visualizations_code

# Load GROQ API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit App Configuration
st.set_page_config(page_title="📊 AI Data Analyst", layout="wide")
st.title("📊 AI-Powered Data Analyst")
st.write("Upload your CSV to generate automated insights and multiple visualizations using Plotly + Groq + LangChain!")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload a CSV File", type=["csv"])

# If file is uploaded
if uploaded_file:
    if not GROQ_API_KEY:
        st.error("❌ Missing GROQ_API_KEY in your .env file.")
    else:
        try:
            # Read CSV into DataFrame
            df = pd.read_csv(uploaded_file)
            st.subheader("🔍 Data Preview")
            st.dataframe(df.head())

            # Prepare head and dtypes for LLM
            df_head = df.head().to_string()
            df_dtypes = df.dtypes.to_string()

            # 🔎 Generate insight report
            with st.spinner("🧠 Generating insight report..."):
                insights_text = generate_insight_report(df_head, df_dtypes)

            # 📋 Key Insights
            st.subheader("📋 Key Insights")

            # Clean and format LLM response into proper markdown bullet points
            raw_insights = insights_text.strip()

            # Ensure bullet points are on new lines
            formatted_insights = raw_insights.replace("•", "\n-")

            # Add markdown header
            formatted_insights = "### 📌 Insights from the Data\n\n" + formatted_insights

            # Display
            st.markdown(formatted_insights)


            # 📊 Generate visualization code
            with st.spinner("📈 Generating visualizations..."):
                vis_code = generate_insights_and_visualizations_code(df_head, df_dtypes)

            # Show generated code
            st.subheader("🧠 Generated Python Code")
            # st.code(vis_code, language='python')

            # Execute the generated visualization code
            st.subheader("📊 Visualizations")
            try:
                local_vars = {"df": df}

                # Safety check
                if "read_csv" in vis_code or "open(" in vis_code:
                    st.error("🚫 Generated code tries to read a file, which is not allowed.")
                else:
                    exec(vis_code, {}, local_vars)

                    figs_found = False
                    for i in range(1, 10):
                        fig = local_vars.get(f"fig{i}")
                        if fig:
                            st.markdown(f"### 📊 Chart {i}")
                            st.plotly_chart(fig, use_container_width=True)
                            figs_found = True

                    if not figs_found:
                        st.warning("⚠️ No `fig1`, `fig2`, etc. objects found in the generated code.")

            except Exception:
                st.error("❌ Error while executing the generated code:")
                st.code(traceback.format_exc())

        except Exception as e:
            st.error("⚠️ Failed to read or process the CSV file.")
            st.code(str(e))

else:
    st.info("📥 Please upload a CSV file to begin.")

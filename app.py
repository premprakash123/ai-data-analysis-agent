import streamlit as st
import google.generativeai as genai
import pandas as pd
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")
st.title("📊 AI Agent: KPI & Forecasting Analyst")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("API Key not found! Please check .streamlit/secrets.toml")
    st.stop() # Stops the app from running if the key is missing

# --- 2. AGENT SETUP ---
# We enable 'code_execution' so Gemini can actually run Python on your data
model = genai.GenerativeModel(
    model_name="gemini-3-flash-preview",
    tools=[{"code_execution": {}}]
)

# --- 3. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload your CSV or JSON file", type=["csv", "json"])

if uploaded_file:
    # Read file for local preview
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_json(uploaded_file)
    
    st.write("### Data Preview", df.head())

    # --- 4. AGENT INTERACTION ---
    query = st.text_area(
        "What would you like to know?",
        placeholder="Perform EDA, calculate KPIs, find peak hours, and forecast next 7 days."
    )

    if st.button("Run Analysis"):
        with st.spinner("Agent is analyzing data and executing code..."):
            
            # Step 4a: Prepare context for Gemini
            # We send the column names and a sample to help it write the right code
            data_context = f"""
            The user has uploaded a dataset.
            Columns: {list(df.columns)}
            Sample Data: {df.head(3).to_dict()}
            
            User Request: {query}
            
            Please:
            1. Use Python (pandas/numpy/matplotlib) to analyze the data.
            2. Identify KPIs and Time Insights (Peak Hours).
            3. Provide a simple forecast for the next 7 periods.
            4. Print the results clearly.
            5.When creating bar charts for cities, ensure ALL unique cities from the 'city' column are included in the plot. Do not truncate the list.
            6. If the user asks for a heatmap, use 'seaborn' and 'matplotlib'. 
            7. Ensure you convert the date column to datetime objects first.
            8. If specific cities like Hyderabad are missing from previous charts, ensure they are included now.
            9. Set the figure size to (12, 6) so it is easy to read in Streamlit.
            10. Always include all cities (like Hyderabad, Pune, Bangalore) in charts unless asked otherwise.
            11. If asked for time patterns, prefer using Heatmaps (Day vs Hour).
            12. When forecasting, use the last 7-14 days of data to project the next week.
            13. Always explain 'why' a certain hour is a peak hour (e.g., 'Lunchtime rush' or 'Evening shopping').
            14. All financial values and KPIs must be displayed in **Indian Rupees (₹)**.
            15. If the data is numerical, format the output using the ₹ symbol (e.g., ₹5,00,000).
            16. In matplotlib/seaborn charts, label the axes as 'Sales in ₹'.
            17. Ensure cities like Hyderabad and Pune are always prioritized in the analysis.
            """

            # Step 4b: Call Gemini
            response = model.generate_content(data_context)
            
            # --- 5. DISPLAY RESULTS ---
            st.markdown("### Analysis Results")
            for part in response.candidates[0].content.parts:
                if part.text:
                    st.write(part.text)
    
                # This part handles the charts the agent creates!
                if part.inline_data:
                    st.image(part.inline_data.data, caption="Analysis Chart")
            
            # Check if there's any code execution output to show
            if hasattr(response.candidates[0].content.parts[0], 'executable_code'):
                with st.expander("View Agent's Python Logic"):
                    st.code(response.candidates[0].content.parts[0].executable_code)

elif "GEMINI_API_KEY" not in st.secrets:
    st.warning("Please enter your Gemini API Key in the sidebar to begin.")
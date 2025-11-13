import os
import streamlit as st
from dotenv import load_dotenv
from dotenv import load_dotenv

load_dotenv()

os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "2000"

print(os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] )
print(os.getenv("OPENAI_API_KEY"))



# Set Streamlit to wide mode
st.set_page_config(layout="wide", page_title="Main Dashboard", page_icon="📊")


data_visualisation_page = st.Page(
    "./Pages/python_visualisation_agent.py", title="Data Visualisation", icon="📈"
)

pg = st.navigation(
    {
        "Visualisation Agent": [data_visualisation_page]
    }
)

pg.run()

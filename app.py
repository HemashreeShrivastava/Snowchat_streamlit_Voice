import os
import io
import pandas as pd
import streamlit as st
import snowflake.connector
from gpt import OpenAIService 
import speech_recognition as sr
import time

# Set wide layout for Streamlit app
st.set_page_config(layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #8BA4E1;
    }
    .sidebar .sidebar-content {
        background-color: #D3D3D3;
    }
    .stTextInput, .stTextArea {
        background-color: #D3D3D3;
        color: black;
        border-radius: 15px;
        border: 2px solid #00bcd4;
        margin: 10px 0;
        padding: 10px;
    }
    .stButton button {
        background-color: #00bcd4;
        color: #ffffff;
        border: 2px solid #0097a7;
        border-radius: 15px;
        margin: 10px 0;
    }
    .stButton button:hover {
        background-color: #0097a7;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #006064;
    }
    /* Center align button */
    .centered-button {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar for selecting between Snowflake and OpenAI settings
with st.sidebar:
    st.title("Settings")
    option = st.selectbox("Choose an option", ["Snowflake Credentials", "OpenAI API Key"])

    if option == "Snowflake Credentials":
        st.caption("Enter Snowflake Credentials:")
        os.environ["SNOWFLAKE_USER"] = st.text_input("User", value="siri")
        os.environ["SNOWFLAKE_PASSWORD"] = st.text_input("Password", value="Techment@123", key="hidden_password", type="password")
        os.environ["SNOWFLAKE_ACCOUNT"] = st.text_input("Account", value="jv51685.central-india.azure")
        os.environ["SNOWFLAKE_WAREHOUSE"] = st.text_input("Warehouse", value="COMPUTE_WH")
        os.environ["SNOWFLAKE_SCHEMA"] = st.text_input("Schema", value="RAW")
        os.environ["SNOWFLAKE_DATABASE"] = st.text_input("Database", value="SALES_REVENUE_DATA")
    elif option == "OpenAI API Key":
        st.caption("Enter OpenAI API Key:")
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            st.info("Please enter OpenAI API Key")
            st.stop()

# Snowflake credentials
creds = {
    "user": os.environ["SNOWFLAKE_USER"],
    "password": os.environ["SNOWFLAKE_PASSWORD"],
    "account": os.environ["SNOWFLAKE_ACCOUNT"],
    "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
    "schema": os.environ["SNOWFLAKE_SCHEMA"],
    "database": os.environ["SNOWFLAKE_DATABASE"],
}

# Caching the Snowflake connection
@st.cache_resource
class SnowflakeDB:
    def __init__(self) -> None:
        self.conn = snowflake.connector.connect(**creds)
        self.cursor = self.conn.cursor()

# Read prompt from file
def read_prompt_file(fname):
    with open(fname, "r") as f:
        return f.read()

# Cache query results
@st.cache_data
def query(_conn, query):
    try:
        return pd.read_sql(query, _conn)
    except Exception as e:
        st.warning("Error in query")
        st.error(e)

# Cache OpenAI response
@st.cache_data
def ask(prompt):
    gpt = OpenAIService()
    response = gpt.prompt(prompt)
    return response["choices"][0]["message"]["content"]

# Get table schemas from Snowflake
def get_tables_schema(_conn):
    with st.expander("View tables schema in database"):
        table_schemas = ""
        df = query(_conn, "show tables")
        st.write(df)
        for table in df["name"]:
            t = f"{os.environ['SNOWFLAKE_DATABASE']}.{os.environ['SNOWFLAKE_SCHEMA']}.{table}"
            ddl_query = f"select get_ddl('table', '{t}');"
            ddl = query(_conn, ddl_query)
            schema = f"\n{ddl.iloc[0, 0]}\n"
            st.write(f"### {table}")
            st.markdown(f"```sql{schema}```")
            st.write("---")
            table_schemas += f"\n{table}\n{schema}\n"
    return table_schemas

# Get dataframe schema information
def df_schema(df):
    sio = io.StringIO()
    df.info(buf=sio)
    df_info = sio.getvalue()
    return df_info

# Validate SQL query for safe execution
def validate_sql(sql):
    restricted_keywords = ["DROP", "ALTER", "TRUNCATE", "UPDATE", "REMOVE"]
    for keyword in restricted_keywords:
        if keyword in sql.upper():
            return False, keyword
    return True, None

# Function to get audio input
def get_audio():
    recognizer = sr.Recognizer()

    # Use the microphone as the audio source
    with sr.Microphone() as source:
        st.write("Listening... Please speak after a few seconds.")
        #time.sleep(1)  
        recognizer.adjust_for_ambient_noise(source)

        try:
            audio = recognizer.listen(source, timeout=3)  
            st.write("Recognizing...")
            text = recognizer.recognize_google(audio)
            st.write("You said: ", text)
            return text  

        except sr.WaitTimeoutError:
            st.write("Listening timed out while waiting for phrase to start")
            return "No speech detected"

        except sr.UnknownValueError:
            st.write("Google Web Speech API could not understand audio")
            return "Could not understand audio"

        except sr.RequestError as e:
            st.write(f"Could not request results from Google Web Speech API; {e}")
            return "API error"

# Main app logic
if __name__ == "__main__":
    st.title("Snowflake-Streamlit Integration")
    st.write("Connect to your Snowflake database and ask questions about your data in real-time.")

    # Initialize the services
    gpt = OpenAIService()
    sf = SnowflakeDB()

    # Get table schemas
    table_schemas = get_tables_schema(sf.conn)
    st.write("---")

    # Layout: Create two columns (one for text area and one for the button)
    col1, col2 = st.columns([3, 1])  # Adjust the ratios as needed

    # Column for text area input
    with col1:
        if 'question' not in st.session_state:
            st.session_state.question = ""

        st.session_state.question = st.text_area(
            "Hi! I’m Snow Chat, your data assistant. Ask me anything about your connected data!",
            value=st.session_state.question,  
            placeholder="What is the total revenue?",
            height=100,
        )

    # Column for the speech-to-text button
    with col2:
        st.markdown('<div class="centered-button">', unsafe_allow_html=True)
        if st.button("🎤 Query by voice"):
            speech_input = get_audio()  
            if speech_input:
                st.session_state.question = speech_input
        st.markdown('</div>', unsafe_allow_html=True)

    # Check if a question has been entered via text or speech
    if st.session_state.question:
        question = st.session_state.question
        
        # Curate the prompt for SQL generation
        prompt = read_prompt_file("sql_prompt.txt")
        prompt = prompt.replace("<<TABLES>>", table_schemas)
        prompt = prompt.replace("<<QUESTION>>", question)
        answer = ask(prompt)

        # Remove any unwanted markdown or formatting characters
        answer = answer.replace("```sql", "").replace("```", "").strip()

        # Validate SQL query
        is_valid, keyword = validate_sql(answer)
        if not is_valid:
            st.error(f"Operation not allowed: {keyword} is restricted!")
            st.stop()

        # Display the generated SQL code
        st.code(answer)

        # Execute the query and display the resulting dataframe
        df = query(sf.conn, answer)
        st.dataframe(df, use_container_width=True)

    # Python code execution section
    question_python = st.text_input(
        "Ask a question about the result",
        placeholder="e.g. Visualize the data",
    )
    if question_python:
        # Curate the prompt for Python
        df_info = df_schema(df)
        prompt = read_prompt_file("python_prompt.txt")
        prompt = prompt.replace("<<DATAFRAME>>", df_info)
        prompt = prompt.replace("<<QUESTION>>", question_python)
        python_code = ask(prompt)

        # Remove any unwanted markdown or formatting characters
        python_code = python_code.replace("```python", "").replace("```", "").strip()

        # Execute Python code safely with error handling
        try:
            exec(python_code)
        except Exception as e:
            st.error(f"Error executing the Python code: {e}")

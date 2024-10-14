import os
import io
import pandas as pd
import streamlit as st
import snowflake.connector
from gpt import OpenAIService
import audio_recorder_streamlit as ar  # Import audio_recorder_streamlit for recording audio
from pydub import AudioSegment
from pydub.playback import play
import tempfile

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

# Function to get audio input using audio_recorder_streamlit
def get_audio():
    audio_bytes = ar.audio_recorder()  # This function records audio directly from the browser

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")  # Play back the recorded audio
        st.write("Audio captured successfully")

        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(audio_bytes)
            audio_path = temp_audio_file.name

        # Use Pydub to convert the audio file for further processing (if needed)
        audio_segment = AudioSegment.from_wav(audio_path)

        # Display the duration of the recorded audio
        st.write(f"Duration of the audio: {len(audio_segment) / 1000:.2f} seconds")
        return audio_path
    else:
        st.write("No audio recorded")
        return None

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
            audio_file_path = get_audio()  
            if audio_file_path:
                # You would need to integrate speech-to-text processing here
                st.session_state.question = "Recognized speech goes here"
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

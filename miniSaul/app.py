import streamlit as st
import json
import base64
from rag import pipeline
import time
import pandas as pd


def get_download_link(file_content, file_name, file_label):
    """
    Create a download link for a file.
    
    Args:
        file_content (str): The content of the file to be downloaded.
        file_name (str): The name of the file to be downloaded.
        file_label (str): The label of the file to be downloaded.
    
    Returns:
        str: A download link for the file.
    """
    b64 = base64.b64encode(file_content.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{file_name}">{file_label}</a>'

def update_log(message, log_messages, log_placeholder):
    """
    Update the log messages and display them in a text area.
    
    Args:
        message (str): The message to be added to the log.
        log_messages (list): The list of log messages.
        log_placeholder (streamlit.empty): The placeholder for the log messages.
    """
    log_messages.append(message)

    log_placeholder.text_area("Processing Logs", value="\n".join(log_messages), height=150, key=f"log_area_{len(log_messages)}")

def app():
    """
    Main function to run the Streamlit app.
    """
    st.title("File Processing App")

    file1 = st.file_uploader("Contract file", type=["docx"])
    file2 = st.file_uploader("Tasks file", type=["xlsx"])

    if file1 and file2:
        if st.button("Process Files"):
            log_placeholder = st.empty()
            log_messages = []
            result = pipeline(file1, file2, lambda message: update_log(message, log_messages, log_placeholder))            
            st.json(json.loads(result))
            st.markdown(get_download_link(result, "checked_tasks.json", "Download Checked Tasks"), unsafe_allow_html=True)

if __name__ == "__main__":
    app()
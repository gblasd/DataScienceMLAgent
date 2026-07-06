import streamlit as st
import requests
import pandas as pd
import io
import time

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="ML Agent Platform",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stApp { margin-top: -20px; }
</style>
""", unsafe_allow_html=True)

# Main layout
st.title("Data Science ML Agent", anchor=False)
st.caption("AI-powered automated machine learning and data analysis platform", text_alignment="left")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files_info" not in st.session_state:
    st.session_state.uploaded_files_info = {}

def check_api_status():
    try:
        # A simple request to see if API is up
        response = requests.get(f"{API_URL}/docs", timeout=1)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

api_online = check_api_status()

# Sidebar for controls and files
with st.sidebar:
    st.header("Workspace", anchor=False, divider=False)
    
    if api_online:
        st.badge("API Online", icon=":material/check_circle:", color="green")
    else:
        st.badge("API Offline", icon=":material/error:", color="red")
        st.error("Please run the backend: `fastapi dev app.py`")
        
    st.space("medium")
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Datasets",
        type=['csv', 'parquet'],
        accept_multiple_files=True,
        help="Upload training data, test data, or any CSV/Parquet files",
        label_visibility="collapsed"
    )
    
    if uploaded_files and api_online:
        if st.button("Upload to Agent", icon=":material/cloud_upload:", use_container_width=False, type="primary"):
            with st.spinner("Uploading..."):
                files = [("files", (f.name, f.getvalue(), "application/octet-stream")) for f in uploaded_files]
                try:
                    res = requests.post(f"{API_URL}/upload", files=files)
                    if res.status_code == 200:
                        st.toast("Files uploaded successfully!", icon=":material/check_circle:")
                        st.session_state.uploaded_files_info.update(res.json().get("files", {}))
                    else:
                        st.error("Failed to upload files.")
                except Exception as e:
                    st.error(f"Error uploading files: {str(e)}")

    if st.session_state.uploaded_files_info:
        with st.expander("Available Files", icon=":material/folder:", expanded=True):
            for filename, info in st.session_state.uploaded_files_info.items():
                size_kb = info.get("size", 0) / 1024
                st.markdown(f"**{filename}**  \n:green-badge[{size_kb:.1f} KB]")
                
    st.space("large")
    st.header("Downloads", anchor=False, divider=False)
    
    col1, col2 = st.columns(2)
    with col1:
        # We can't check file existence directly since it's on server, we try to fetch it
        st.link_button("Model", f"{API_URL}/download/model", icon=":material/download:", use_container_width=True)
    with col2:
        st.link_button("Predictions", f"{API_URL}/download/predictions", icon=":material/download:", use_container_width=True)

# Main chat interface
left_col, right_col = st.columns([7, 3])

with right_col:
    with st.expander("💡 Example Commands", icon=":material/lightbulb:", expanded=True):
        st.markdown("""
        **Data Loading:**
        `load_dataset(path, target)`
        `set_target(column)`
        `describe_data()`
        `preview_data(rows?)`

        **Training & Optimization:**
        `train_classification()`
        `train_regression()`
        `optimize_logistic(trials?)`
        `optimize_forest_regressor(trials?)`

        **Predictions & Results:**
        `predict(test_data_path)`
        `show_best_model()`
        `show_history()`
        """)
        
    st.space("medium")
    if st.button("Clear History", icon=":material/delete:", use_container_width=True):
        if api_online:
            requests.post(f"{API_URL}/clear")
            st.session_state.messages = []
            st.rerun()

with left_col:
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user" and "paraphrased" in message:
                st.markdown(f"**Original:** {message['content']}")
                st.markdown(f"**Understood as:** _{message['paraphrased']}_")
            else:
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your data or give ML commands...", disabled=not api_online):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(f"**Original:** {prompt}")
            
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    res = requests.post(f"{API_URL}/chat", json={"message": prompt})
                    if res.status_code == 200:
                        data = res.json()
                        response = data.get("response", "")
                        paraphrased = data.get("paraphrased", prompt)
                        
                        st.markdown(response)
                        
                        # Update session state with the paraphrased version and response
                        st.session_state.messages[-1]["paraphrased"] = paraphrased
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        error_msg = f"API Error: {res.status_code}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except requests.exceptions.RequestException as e:
                    error_msg = f"Connection error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.rerun()

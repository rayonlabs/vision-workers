#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import json
import numpy as np
import sys
import os
from datetime import datetime
import importlib.util
import inspect

# Import the test_validation.py module
spec = importlib.util.spec_from_file_location("test_validation", "test_validation.py")
test_validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_validation)

# Page configuration
st.set_page_config(
    page_title="LLM Testing Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'test_results' not in st.session_state:
    st.session_state.test_results = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'current_results_df' not in st.session_state:
    st.session_state.current_results_df = None

# Add log message with immediate UI update
def add_log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.log_messages.append(log_entry)
    
    # Update the logs display - using a unique key for the text area
    if 'log_area' in st.session_state:
        st.session_state.log_area.text(
            "\n".join(st.session_state.log_messages)
        )
    
    # Also print to console for debugging
    print(log_entry)

# Function to clear all session state and UI elements
def clear_all():
    st.session_state.test_results = None
    st.session_state.progress = 0
    st.session_state.log_messages = []
    st.session_state.current_results_df = None
    
# Header
st.title("LLM Testing Dashboard")
st.markdown("Configure and run tests for LLM models using miner and validator endpoints.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Server configuration
    st.subheader("Server Configuration")
    miner_base_url = st.text_input("Miner Base URL", value=test_validation.MINER_BASE_URL)
    validator_base_url = st.text_input("Validator Base URL", value=test_validation.VALIDATOR_BASE_URL)
    
    # Task configuration
    st.subheader("Task Configuration")
    task_type = st.selectbox("Task Type", ["chat", "completion"], 
                            index=0 if test_validation.TASK_TYPE == "chat" else 1)
    task_id = st.text_input("Task ID", value=test_validation.TASK_ID)
    
    # Test configuration
    st.subheader("Test Configuration")
    n_data_points = st.number_input("Number of Test Points", min_value=1, value=test_validation.N_DATA_POINTS)
    
    # Text generation parameters
    st.subheader("Text Generation Parameters")
    col1, col2 = st.columns(2)
    with col1:
        min_words_message = st.number_input("Min Words Per Message", 
                                         min_value=5, value=test_validation.MIN_WORDS_PER_MESSAGE)
        min_chat_messages = st.number_input("Min Chat Messages", 
                                         min_value=2, value=test_validation.MIN_CHAT_MESSAGES)
        completion_min_words = st.number_input("Min Completion Words", 
                                            min_value=10, value=test_validation.COMPLETION_MIN_WORDS)
    with col2:
        max_words_message = st.number_input("Max Words Per Message", 
                                         min_value=10, value=test_validation.MAX_WORDS_PER_MESSAGE)
        max_chat_messages = st.number_input("Max Chat Messages", 
                                         min_value=3, value=test_validation.MAX_CHAT_MESSAGES)
        completion_max_words = st.number_input("Max Completion Words", 
                                            min_value=20, value=test_validation.COMPLETION_MAX_WORDS)
    
    # Temperature distribution
    st.subheader("Temperature Distribution")
    temp_mean = st.slider("Mean Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    temp_std = st.slider("Temperature Std Dev", min_value=0.05, max_value=0.5, value=0.25, step=0.05)
    
    # Advanced model configuration
    with st.expander("Advanced Model Configuration"):
        model_name = st.text_input("Model Name", value=test_validation.MODEL_CONFIG["model"])
        tokenizer_name = st.text_input("Tokenizer Name", value=test_validation.MODEL_CONFIG["tokenizer"])
        half_precision = st.checkbox("Half Precision", value=test_validation.MODEL_CONFIG["half_precision"])
        tensor_parallel_size = st.number_input("Tensor Parallel Size", 
                                            min_value=1, max_value=8, 
                                            value=test_validation.MODEL_CONFIG["tensor_parallel_size"])
        gpu_memory_utilization = st.slider("GPU Memory Utilization", 
                                        min_value=0.1, max_value=1.0, 
                                        value=test_validation.MODEL_CONFIG["gpu_memory_utilization"], 
                                        step=0.1)
        max_model_len = st.number_input("Max Model Length", 
                                     value=test_validation.MODEL_CONFIG["max_model_len"], 
                                     min_value=1024, max_value=128000)
        eos_token_id = st.number_input("EOS Token ID", 
                                    value=test_validation.MODEL_CONFIG["eos_token_id"])

# Create a container for status messages
status_container = st.container()

# Main layout with fixed containers
col1, col2 = st.columns([3, 2])

with col1:
    # Test control panel
    st.subheader("Test Control")
    button_cols = st.columns([1, 1])
    with button_cols[0]:
        start_button = st.button("Start Test", disabled=st.session_state.is_running)
    with button_cols[1]:
        clear_button = st.button("Clear All", disabled=st.session_state.is_running)
    
    # Results section with fixed containers
    st.subheader("Test Results")
    results_table = st.empty()  # For dataframe
    scatter_plot = st.empty()   # For scatter plot
    # Removed histogram plot

with col2:
    # Progress section
    st.subheader("Test Progress")
    progress_bar = st.empty()  # For progress bar
    
    # Log section
    st.subheader("Execution Log")
    log_container = st.container(height=500)
    with log_container:
        st.session_state.log_area = st.empty()
        st.session_state.log_area.text("")

# Custom run function that updates Streamlit
async def run_streamlit_test():
    try:
        # Reset state
        st.session_state.progress = 0
        st.session_state.log_messages = []
        st.session_state.current_results_df = None
        
        # Clear UI elements
        results_table.empty()
        scatter_plot.empty()
        progress_bar.progress(0)
        st.session_state.log_area.text("")
        
        # Override the module variables with our UI values
        test_validation.MINER_BASE_URL = miner_base_url
        test_validation.VALIDATOR_BASE_URL = validator_base_url
        test_validation.TASK_TYPE = task_type
        test_validation.TASK_ID = task_id
        test_validation.N_DATA_POINTS = n_data_points
        test_validation.MIN_WORDS_PER_MESSAGE = min_words_message
        test_validation.MAX_WORDS_PER_MESSAGE = max_words_message
        test_validation.MIN_CHAT_MESSAGES = min_chat_messages
        test_validation.MAX_CHAT_MESSAGES = max_chat_messages
        test_validation.COMPLETION_MIN_WORDS = completion_min_words
        test_validation.COMPLETION_MAX_WORDS = completion_max_words
        
        # Update model config
        test_validation.MODEL_CONFIG = {
            "model": model_name,
            "tokenizer": tokenizer_name,
            "half_precision": half_precision,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "eos_token_id": eos_token_id
        }
        
        add_log(f"Starting test with {n_data_points} data points")
        
        # For single test, just run it directly
        if n_data_points <= 1:
            add_log("Running single test...")
            
            # We need to create a new session for each run
            async with test_validation.aiohttp.ClientSession() as session:
                try:
                    result = await test_validation.run_single_test(session)
                    
                    # Show results
                    add_log(f"Test completed with score: {result['score']}")
                    
                    # For chat vs completion content info
                    if task_type == "chat":
                        message_count = len(result["payload"]["messages"])
                        content_info = f"{message_count} messages"
                    else:
                        word_count = len(result["payload"]["prompt"].split())
                        content_info = f"{word_count} words"
                    
                    # Create a dataframe with single result
                    df = pd.DataFrame([{
                        "temperature": result["temperature"],
                        "score": result["score"],
                        "content_info": content_info
                    }])
                    st.session_state.current_results_df = df
                    
                    # Update the results UI
                    results_table.dataframe(df[['temperature', 'score', 'content_info']], use_container_width=True)
                    
                    # Only show plots if we have valid scores
                    if result["score"] is not None:
                        fig = px.scatter(df, x='temperature', y='score', 
                                        title='Temperature vs Score',
                                        labels={'temperature': 'Temperature', 'score': 'Score'})
                        fig.update_layout(height=400)
                        scatter_plot.plotly_chart(fig, use_container_width=True)
                    
                    # Add validator result details to log
                    if result["validator_result"]:
                        add_log("Validator result received")
                        add_log(f"Validator score: {result['score']}")
                    
                    # Update progress
                    st.session_state.progress = 1.0
                    progress_bar.progress(1.0)
                    
                except Exception as e:
                    add_log(f"Error: {str(e)}")
                    raise e
        
        # For multiple tests, we need to mimic run_full_test but with progress updates
        else:
            add_log(f"Running full test with {n_data_points} data points...")
            
            # Generate temperatures with normal distribution
            temps = []
            while len(temps) < n_data_points:
                t = np.random.normal(temp_mean, temp_std)
                if 0 <= t <= 1:
                    temps.append(t)
            
            results = []
            async with test_validation.aiohttp.ClientSession() as session:
                for i, temperature in enumerate(temps):
                    progress = (i) / n_data_points
                    st.session_state.progress = progress
                    
                    # Update progress bar
                    progress_bar.progress(progress)
                    
                    add_log(f"Running test {i+1}/{n_data_points} with temperature {temperature:.4f}")
                    
                    try:
                        result = await test_validation.run_single_test(session, temperature)
                        
                        # Extract message/word count
                        if task_type == "chat":
                            message_count = len(result["payload"]["messages"])
                            content_info = f"{message_count} messages"
                        else:
                            word_count = len(result["payload"]["prompt"].split())
                            content_info = f"{word_count} words"
                        
                        results.append({
                            "temperature": temperature,
                            "score": result["score"],
                            "content_info": content_info,
                            "payload": json.dumps(result["payload"]),
                            "response": json.dumps(result["miner_response"]),
                        })
                        
                        add_log(f"Test {i+1} completed. Score: {result['score']} | {content_info}")
                        
                        # Update and display current results after each test
                        temp_df = pd.DataFrame(results)
                        st.session_state.current_results_df = temp_df
                        
                        # Update each UI element in place
                        results_table.dataframe(temp_df[['temperature', 'score', 'content_info']], use_container_width=True)
                        
                        # Only show plots if we have valid scores
                        if 'score' in temp_df.columns and not temp_df['score'].isna().all():
                            # Temperature vs Score plot
                            fig = px.scatter(temp_df, x='temperature', y='score', 
                                            title='Temperature vs Score',
                                            labels={'temperature': 'Temperature', 'score': 'Score'})
                            fig.update_layout(height=400)
                            scatter_plot.plotly_chart(fig, use_container_width=True)
                            
                            # Removed score distribution histogram
                        
                    except Exception as e:
                        add_log(f"Error in test {i+1}: {str(e)}")
                        results.append({
                            "temperature": temperature,
                            "score": None,
                            "content_info": None,
                            "payload": None,
                            "response": None,
                            "error": str(e)
                        })
            
            # Removed CSV saving code
            
            # Final progress update
            st.session_state.progress = 1.0
            progress_bar.progress(1.0)

        return True
    
    except Exception as e:
        add_log(f"Error in test execution: {str(e)}")
        progress_bar.progress(0)
        results_table.error(f"Test failed: {str(e)}")
        raise e

# Run the test when button is clicked
if start_button:
    st.session_state.is_running = True
    try:
        # Clear any previous status messages
        status_container.empty()
        
        # Run the test
        success = asyncio.run(run_streamlit_test())
        
        # Only show success message if everything completes properly
        if success:
            # Show success message
            status_container.success("Test completed successfully!")
            
            # Add final log message
            add_log("Test completed successfully!")
            
    except Exception as e:
        error_msg = f"Test failed: {str(e)}"
        status_container.error(error_msg)
        add_log(error_msg)
    finally:
        st.session_state.is_running = False

# Handle Clear button
if clear_button:
    # Clear session state
    clear_all()
    
    # Clear UI elements
    results_table.empty()
    scatter_plot.empty()
    progress_bar.empty()
    st.session_state.log_area.text("")
    
    # Show info message
    results_table.info("No test results to display. Start a test to see results here.")
    
    # Clear status
    status_container.empty()

# Instructions at the bottom
with st.expander("How to use this dashboard"):
    st.markdown("""
    ### Instructions
    
    1. **Configure the test parameters** in the sidebar
    2. **Start the test** by clicking the "Start Test" button
    3. **Monitor progress** in the log section
    4. **View results** in the results section
    5. **Clear All** to reset the dashboard and start over
    
    ### Test Types
    
    - **Single Test (N=1)**: Runs one test with the specified parameters
    - **Full Test (N>1)**: Runs N tests with temperatures following a normal distribution
    """)
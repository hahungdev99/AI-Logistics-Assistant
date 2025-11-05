"""
Hermes AI Logistics Assistant - Streamlit Frontend
Simple chat interface for interacting with the AI agent
"""

import streamlit as st
import sys
from pathlib import Path
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from backend.agent import create_hermes_agent

# Page configuration
st.set_page_config(
    page_title="Hermes AI Logistics Assistant",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
    }
    .stat-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    .example-query {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.3rem 0;
        cursor: pointer;
        border: 1px solid #ffc107;
    }
    .example-query:hover {
        background-color: #ffe69c;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "agent_initialized" not in st.session_state:
    st.session_state.agent_initialized = False

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Enter your OpenAI API key"
    )
    
    # Data file path
    data_path = st.text_input(
        "Data File Path",
        value="data/shipments.csv",
        help="Path to shipments CSV file"
    )
    
    # Initialize button
    if st.button("🚀 Initialize Agent", type="primary"):
        if not api_key:
            st.error("Please enter your OpenAI API key!")
        else:
            with st.spinner("Initializing Hermes Agent..."):
                try:
                    st.session_state.agent = create_hermes_agent(api_key, data_path)
                    st.session_state.agent_initialized = True
                    st.success("✓ Agent initialized successfully!")
                    
                    # Show data summary
                    summary = st.session_state.agent.get_data_summary()
                    st.markdown("### 📊 Data Summary")
                    st.json(summary)
                    
                except Exception as e:
                    st.error(f"Error initializing agent: {e}")
                    st.session_state.agent_initialized = False
    
    # Clear conversation button
    if st.session_state.agent_initialized:
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.agent.reset_memory()
            st.success("Conversation cleared!")
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 📚 Example Queries")
    
    example_queries = [
        "Which route had the most delays last week?",
        "Show the top 3 warehouses with highest processing time",
        "What was the average delay in October?",
        "Predict the average delay for next week",
        "How many shipments were delayed in October?",
        "List all shipments delayed due to Weather last week",
        "What is the on-time delivery rate this week?",
        "Show delay trends for last week - increasing or decreasing?",
        "Give me warehouse optimization recommendations",
        "What's the average processing time at WH1?"
    ]
    
    for query in example_queries:
        if st.button(query, key=f"ex_{query[:20]}", use_container_width=True):
            if not st.session_state.agent_initialized:
                st.warning("Please initialize the agent first!")
            else:
                # Add to chat
                st.session_state.messages.append({"role": "user", "content": query})
                with st.spinner("Processing..."):
                    response = st.session_state.agent.chat(query)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["response"]
                    })
                st.rerun()

# Main content
st.markdown('<div class="main-header">📦 Hermes AI Logistics Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your intelligent partner for logistics data analysis and predictions</div>', unsafe_allow_html=True)

# Show initialization message if not initialized
if not st.session_state.agent_initialized:
    st.info("👈 Please initialize the agent in the sidebar to start chatting!")
    
    # Show features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <h3>🔍 Data Analysis</h3>
            <p>Query shipment data with natural language</p>
            <ul>
                <li>Route performance</li>
                <li>Warehouse metrics</li>
                <li>Delay analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <h3>📈 Trend Detection</h3>
            <p>Identify patterns in your logistics data</p>
            <ul>
                <li>Time-series analysis</li>
                <li>Performance trends</li>
                <li>Delay patterns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <h3>🔮 Predictions</h3>
            <p>Forecast future logistics metrics</p>
            <ul>
                <li>Delay predictions</li>
                <li>ML-based forecasts</li>
                <li>Optimization tips</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

else:
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>👤 You:</strong><br/>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <strong>🤖 Hermes:</strong><br/>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Ask me about your logistics data...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get agent response
        with st.spinner("🤔 Analyzing your request..."):
            response = st.session_state.agent.chat(user_input)
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["response"]
            })
        
        # Rerun to display new messages
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    Hermes AI Logistics Assistant | Powered by LangChain & GPT-4o | Made with ❤️ for Operations Managers
</div>
""", unsafe_allow_html=True)
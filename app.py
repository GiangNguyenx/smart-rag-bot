import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_classic.agents.agent_types import AgentType
from langchain_community.callbacks import StreamlitCallbackHandler

from langchain_core.messages import HumanMessage

from src.ds_tools import analyze_clusters, predict_trend
from src.graph_agent import build_data_analyst_graph

# 1. Cấu hình trang
st.set_page_config(page_title="AI Data Scientist Agent", page_icon="🧪", layout="wide")
st.title("🧪 AI Data Scientist - Tích hợp Custom Tools")

load_dotenv() 

if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph_messages" not in st.session_state:
    st.session_state.graph_messages = []

# 2. Setup Sidebar & API Key
with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    # Lấy API Key (Ưu tiên từ .env, nếu không có thì nhập tay)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = st.text_input("Nhập Groq API Key", type="password")
    
    st.divider()
    
    # Upload File
    uploaded_file = st.file_uploader("📂 Upload file CSV của bạn", type=["csv"])
    
    st.info("💡 Hãy upload file có tiêu đề cột tiếng Anh hoặc tiếng Việt không dấu để AI hiểu tốt nhất.")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.graph_messages = []
        st.rerun()

def main():
    if uploaded_file is not None and api_key:
        df = pd.read_csv(uploaded_file)
        
        # Show dataframe preview
        with st.expander("📊 Xem trước dữ liệu"):
            st.dataframe(df.head())
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
        
        # Build the graph agent
        graph_app = build_data_analyst_graph(df, api_key)
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Hỏi AI về dữ liệu của bạn..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("🤖 LangGraph đang phân tích..."):
                    try:
                        # Add user message to graph history
                        st.session_state.graph_messages.append(HumanMessage(content=prompt))
                        
                        # Prepare input for LangGraph with full history
                        inputs = {"messages": st.session_state.graph_messages}
                        
                        # Invoke the graph
                        result = graph_app.invoke(inputs, config={"recursion_limit": 15})
                        
                        # Update graph history with all new messages
                        st.session_state.graph_messages = result["messages"]
                        
                        # Extract the final AI message
                        last_msg = result["messages"][-1]
                        final_response = last_msg.content
                        
                        # Display response
                        st.write(final_response)
                        
                        # Check if matplotlib generated any plots
                        if plt.gcf().get_axes():
                            st.pyplot(plt.gcf())
                            plt.clf()  # Clear the figure for next plot
                        
                        # Save assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": final_response})
                    
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    elif not api_key:
        st.warning("⚠️ Vui lòng nhập GROQ API Key trong sidebar")
    else:
        st.info("📁 Vui lòng upload file CSV để bắt đầu")

if __name__ == "__main__":
    main()
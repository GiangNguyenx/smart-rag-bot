import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# 1. Cấu hình trang
st.set_page_config(page_title="AI Data Analyst 📊", layout="wide")
st.title("📊 AI Phân tích Dữ liệu (CSV/Excel)")

# 2. Setup Sidebar & API Key
with st.sidebar:
    st.header("Cấu hình")
    # Để deploy lên mạng, ta sẽ lấy key từ Secrets của Streamlit (sẽ hướng dẫn ở Phần 2)
    # Nhưng khi chạy local, nó sẽ lấy từ file .env hoặc nhập tay
    api_key = st.text_input("Nhập Groq API Key (Nếu chưa set trong Secrets)", type="password")
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload file CSV của bạn", type=["csv"])

# 3. Logic chính
def main():
    # Kiểm tra API Key
    if not api_key and "GROQ_API_KEY" not in os.environ:
        st.warning("Vui lòng nhập API Key để bắt đầu.")
        return
    
    final_key = api_key if api_key else os.environ["GROQ_API_KEY"]

    if uploaded_file is not None:
        # Load dữ liệu vào Pandas DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Hiện bảng dữ liệu (Preview)
        st.write("### Dữ liệu của bạn:")
        st.dataframe(df.head())

        # KHỞI TẠO AI AGENT
        # Đây là "bộ não" biết code Python để trả lời câu hỏi về data
        llm = ChatGroq(
            model_name="llama3-8b-8192", 
            api_key=final_key,
            temperature=0 # Data cần chính xác, không sáng tạo
        )

        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True, 
            allow_dangerous_code=True, # Cần thiết để AI chạy lệnh Python tính toán
            handle_parsing_errors=True # Tự sửa lỗi nếu code sai
        )

        # Giao diện Chat
        st.write("### 🤖 Chat với dữ liệu")
        query = st.text_input("Hỏi gì về bảng dữ liệu này đi (VD: Tổng doanh thu theo từng tháng?)")

        if st.button("Phân tích ngay"):
            with st.spinner("AI đang đọc dữ liệu và tính toán..."):
                try:
                    response = agent.run(query)
                    st.success("Kết quả:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Lỗi rồi: {e}")

if __name__ == "__main__":
    main()
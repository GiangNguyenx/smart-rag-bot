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

from src.ds_tools import analyze_clusters, predict_trend

# 1. Cấu hình trang
st.set_page_config(page_title="AI Data Scientist Agent", page_icon="🧪", layout="wide")
st.title("🧪 AI Data Scientist - Tích hợp Custom Tools")

load_dotenv() 

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
    
    st.info(" Hãy upload file có tiêu đề cột tiếng Anh hoặc tiếng Việt không dấu để AI hiểu tốt nhất.")

def main():
    if uploaded_file is not None and api_key:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        # 1. SETUP LLM
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=api_key, temperature=0)

        # 2. CHUẨN BỊ PROMPT ĐỂ DẠY AGENT VỀ TOOLS
        prefix_prompt = """
        Bạn là một Chuyên gia Data Scientist chuyên nghiệp.
        Bạn đang làm việc với một DataFrame pandas tên là `df`.

        Bạn có sẵn các HÀM CUSTOM mạnh mẽ nằm trong module `src.ds_tools`.
        ĐỂ SỬ DỤNG CHÚNG, BẠN BẮT BUỘC PHẢI IMPORT TRƯỚC KHI GỌI.

        Danh sách hàm và cách import:
        
        1. Hàm Phân cụm (Clustering):
           - Cách dùng:
             ```python
             from src.ds_tools import analyze_clusters
             analyze_clusters(df, features=['Age', 'Spending_Score'], n_clusters=3)
             ```
        
        2. Hàm Dự báo (Forecasting):
           - Cách dùng:
             ```python
             from src.ds_tools import predict_trend
             predict_trend(df, target_col='Revenue', months_ahead=5)
             ```

        QUY TẮC QUAN TRỌNG:
        - Luôn luôn viết dòng `from src.ds_tools import ...` ở đầu đoạn code bạn tạo ra.
        - Không được tự bịa ra code K-Means hay LinearRegression mới, hãy dùng hàm đã import.
        - `df` đã có sẵn, không cần load lại file csv.
        """

        # 3. TẠO AGENT
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            prefix=prefix_prompt 
        )

        # 4. CHAT LOOP
        if prompt := st.chat_input("VD: Dự báo Sales 3 tháng tới / Phân cụm KH theo Age và Score"):
            st.chat_message("user").write(prompt)
            
            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                try:
                    # Để agent chạy được hàm custom, ta phải đưa hàm đó vào namespace (phạm vi biến)
                    # Cách "hack" nhẹ để Pandas Agent nhìn thấy hàm của chúng ta:
                    # Ta gán hàm vào biến toàn cục hoặc truyền vào input (tùy version langchain).
                    # Nhưng với Streamlit, cách đơn giản nhất là agent sẽ dùng `eval()` hoặc `exec()`.
                    # Để an toàn và hiệu quả, ta dùng tham số `extra_tools` (nếu dùng OpenAI Functions)
                    # hoặc đơn giản là để hàm có sẵn trong global scope của file này.
                    
                    # Bước quan trọng: Đảm bảo matplotlib clean trước khi vẽ
                    plt.clf()
                    
                    response = agent.invoke(
                        {"input": prompt},
                        config={"callbacks": [st_callback]}
                    )
                    st.write(response["output"])
                    
                    # Hiển thị biểu đồ nếu Custom Tool đã vẽ
                    if plt.gcf().get_axes():
                        st.pyplot(plt)
                        st.success("Biểu đồ được tạo bởi Custom DS Tool 🧪")
                        
                except Exception as e:
                    st.error(f"Lỗi: {e}")

if __name__ == "__main__":
    main()
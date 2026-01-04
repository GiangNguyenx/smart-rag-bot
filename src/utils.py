import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

# Hàm load model và embeddings (Chạy 1 lần dùng mãi)
def get_llm_resources():
    llm = ChatGroq(model_name="llama-3.3-70b-versatile")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

# Hàm xử lý file PDF người dùng upload
def process_pdf(uploaded_file):
    # 1. Lưu file tạm thời vào thư mục data
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 2. Load và cắt nhỏ file
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_documents = text_splitter.split_documents(docs)
    
    # 3. Xóa file tạm (Dọn rác - Optional)
    os.remove(file_path)
    
    return final_documents

# Hàm tạo Chain RAG
def create_rag_chain(final_documents):
    llm, embeddings = get_llm_resources()
    
    # Tạo Vector Store
    vector_db = FAISS.from_documents(final_documents, embeddings)
    retriever = vector_db.as_retriever()
    
    # Tạo Prompt
    prompt = ChatPromptTemplate.from_template("""
    Bạn là trợ lý AI thông minh. Hãy trả lời câu hỏi dựa trên context sau:
    <context>
    {context}
    </context>
    Câu hỏi: {input}
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain
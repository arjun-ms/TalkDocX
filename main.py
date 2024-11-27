import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

class PDFChatApp:
    def __init__(self):
        # Initialize OpenAI embeddings and chat model
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
        
        # Initialize session state variables
        if 'conversation' not in st.session_state:
            st.session_state.conversation = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
    def process_pdf(self, pdf_file):
        # Load PDF and split into chunks
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        
        # Create conversation chain
        memory = ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True
        )
        
        st.session_state.conversation = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        
        st.success("PDF processed successfully!")
    
    def handle_user_input(self, user_question):
        if st.session_state.conversation:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
            
            # Display conversation
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(f"🧑‍🦰 Human: {message.content}")
                else:
                    st.write(f"🤖 AI: {message.content}")
        
    def main(self):
        st.set_page_config(page_title="PDF Chat", page_icon=":books:")
        st.header("Chat with your PDF 💬")
        
        # PDF Upload
        pdf_file = st.file_uploader(
            "Upload your PDF", 
            type=['pdf'], 
            help="Upload a PDF file to start chatting"
        )
        
        # Process uploaded PDF
        if pdf_file is not None:
            self.process_pdf(pdf_file)
        
        # User input section
        user_question = st.text_input("Ask a question about your PDF:")
        
        if user_question:
            self.handle_user_input(user_question)

def main():
    app = PDFChatApp()
    app.main()

if __name__ == "__main__":
    main()
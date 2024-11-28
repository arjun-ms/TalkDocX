import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Langchain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint as HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# from langchain.vectorstores import FAISS
# from langchain.llms import HuggingFaceHub

# Load environment variables
load_dotenv()

class PDFChatApp:
    def __init__(self):
        # Validate Hugging Face Token
        self.huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
        if not self.huggingface_token:
            st.error("Hugging Face Token is missing. Please set it in .env file.")
            return

        # Initialize embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(
                # trying out nvidia's embedding model
                model_name="BAAI/bge-en-icl"
            )
        except Exception as e:
            st.error(f"Embedding initialization error: {e}")
            return

        ## Initialize Qwen2.5-72B-Instruct model
        try:
            self.llm = HuggingFaceHub(
                repo_id="Qwen/Qwen2.5-72B-Instruct",
                temperature=0.2,  # For more deterministic answers
                max_new_tokens= 512,  # Reduced max_new_tokens for concise responses
                huggingfacehub_api_token=self.huggingface_token,
                model_kwargs={                       # Pass model-specific parameters here
                'max_length': 700,                 # Reduced max_length for quicker responses
                }
            )
        except Exception as e:
            st.error(f"LLM initialization error: {e}")
            return
        
        # Custom prompt template
        self.prompt_template = """
        Context: {context}

        User Question: {question}

        Based on the provided context, please answer the question comprehensively and precisely. 
        If the answer is not directly available in the context, 
        explain why you cannot provide a definitive answer.
        
        Helpful Answer:"""

        self.PROMPT = PromptTemplate(
            template=self.prompt_template, 
            input_variables=["context", "question"]
        )

        # Initialize session state variables
        if 'conversation' not in st.session_state:
            st.session_state.conversation = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Clear session on restart (Reset chat history)
        if 'clear_chat_history' in st.session_state and st.session_state.clear_chat_history:
            st.session_state.chat_history = []
            st.session_state.clear_chat_history = False

    def process_pdf(self, pdf_file):
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(pdf_file.getvalue())
            tmpfile_path = tmpfile.name
        
        try:
            # Load PDF
            loader = PyPDFLoader(tmpfile_path)
            documents = loader.load()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )
            texts = text_splitter.split_documents(documents)
            
            # Create vector store
            vectorstore = FAISS.from_documents(texts, self.embeddings)
            
            # Conversation memory
            memory = ConversationBufferMemory(
                memory_key='chat_history', 
                return_messages=True
            )
            
            # Create conversation chain
            st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=vectorstore.as_retriever(),
                memory=memory,
                combine_docs_chain_kwargs={'prompt': self.PROMPT}
            )
            
            st.success("PDF processed successfully!")
        
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
        
        finally:
            # Clean up temporary file
            os.unlink(tmpfile_path)
    
    def handle_user_input(self, user_question):
        if st.session_state.conversation:
            try:
                # Get the response from the model
                response = st.session_state.conversation({'question': user_question})
                st.session_state.chat_history = response['chat_history']
                
                # Filter and display only the relevant output (Helpful Answer)
                for message in st.session_state.chat_history:
                    print("=====")
                    print(message)
                    print("=====")
                    st.write(message.content)  
                    # if "Helpful Answer:" in message.content:  # Check for the helpful answer part
                    #     st.write(message.content)  # Only show helpful answer
                    #     break  # Exit after displaying the helpful answer

            except Exception as e:
                st.error(f"Error processing question: {e}")
        
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

        # Button to clear chat history (optional, if you want manual control)
        if st.button("Clear Chat History"):
            st.session_state.clear_chat_history = True
            st.write("Chat history cleared!")

def main():
    app = PDFChatApp()
    app.main()

if __name__ == "__main__":
    main()

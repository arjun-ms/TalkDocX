import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import misc.main as main
from sentence_transformers import SentenceTransformer

# Langchain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()  # Ensure that this is called to load .env

# Function to initialize OpenAI API key
def initialize_openai():
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        st.error("OpenAI API Key is missing. Please set it in .env file.")
        return None
    main.api_key = openai_api_key
    return openai_api_key

# Function to initialize the OpenAI model
def initialize_llm(openai_api_key):
    try:
        llm = OpenAI(
            openai_api_key=openai_api_key,
            temperature=0.1, 
            max_tokens=512  # Set the max tokens for concise responses
        )
        return llm
    except Exception as e:
        st.error(f"LLM initialization error: {e}")
        return None

# Function to generate embeddings using SentenceTransformer
def generate_embeddings(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the pre-trained model
    embeddings = model.encode(texts)  # Create embeddings for the list of texts
    return embeddings  # This returns a NumPy array of embeddings

# Process PDF and generate embeddings
def process_pdf(pdf_file):
    openai_api_key = initialize_openai()
    if openai_api_key is None:
        return None, None

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(pdf_file.getvalue())
        tmpfile_path = tmpfile.name
    
    try:
        # Load the PDF and extract text
        loader = PyPDFLoader(tmpfile_path)
        documents = loader.load()
        
        # Split the document into smaller chunks of text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        
        # Extract the content (texts) from the documents
        text_contents = [text.page_content for text in texts]

        # Generate embeddings for each chunk of text
        embeddings = generate_embeddings(text_contents)
        
        # Ensure embeddings are in the correct format (NumPy arrays)
        embeddings = embeddings.tolist()  # Convert the embeddings to list of lists

        print("\n=======embeddings start=======\n")
        print(embeddings)
        print("\n=======embeddings end=======\n")

        # Create FAISS vector store from documents and embeddings
        faiss_vectorstore = FAISS.from_documents(
            documents=texts,  # List of Document objects
            embedding=embeddings  # Embeddings are now correctly passed
        )
        
        # Set up conversation memory
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        return faiss_vectorstore, memory
    
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, None
    
    finally:
        # Clean up temporary file
        os.unlink(tmpfile_path)

# Function to handle user input and get the response
def handle_user_input(user_question, conversation_chain):
    if conversation_chain:
        try:
            response = conversation_chain({'question': user_question})
            chat_history = response['chat_history']
            
            # Filter only relevant output (e.g., helpful answer)
            for message in chat_history:
                if message.content.startswith("Helpful Answer:"):
                    st.write(message.content)  # Only show helpful answer
                    break  # Exit after displaying the helpful answer

        except Exception as e:
            st.error(f"Error processing question: {e}")

# Function to run the main app
def run_app():
    st.set_page_config(page_title="PDF Chat", page_icon=":books:")
    st.header("Chat with your PDF 💬")
    
    # Initialize OpenAI
    openai_api_key = initialize_openai()
    if not openai_api_key:
        return
    
    # Initialize OpenAI model
    llm = initialize_llm(openai_api_key)
    if not llm:
        return

    # Custom prompt template for OpenAI
    prompt_template = """
    Context: {context}

    User Question: {question}

    Based on the provided context, please answer the question comprehensively and precisely. 
    If the answer is not directly available in the context, 
    explain why you cannot provide a definitive answer.
    
    Helpful Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # PDF Upload
    pdf_file = st.file_uploader(
        "Upload your PDF", 
        type=['pdf'], 
        help="Upload a PDF file to start chatting"
    )

    if pdf_file is not None:
        # Process uploaded PDF and get vectorstore and memory
        vectorstore, memory = process_pdf(pdf_file)
        
        if vectorstore and memory:
            # Create conversation chain
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory,
                combine_docs_chain_kwargs={'prompt': PROMPT}
            )
            st.session_state.conversation = conversation_chain
            st.success("PDF processed successfully!")

    # User input section
    user_question = st.text_input("Ask a question about your PDF:")
    
    if user_question:
        handle_user_input(user_question, st.session_state.conversation)

if __name__ == "__main__":
    run_app()

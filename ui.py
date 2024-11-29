import os
import streamlit as st
import pdfplumber
from langchain.schema import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from io import BytesIO

#! modules for speech input
import speech_recognition as sr
import pyttsx3
import threading

# Load environment variables
load_dotenv()

# Setting Up the Environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    

# Streamlit App Setup
st.title("PDF Query with Google Gemini")

# Initialize the Speech-to-Text and Text-to-Speech engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Function to convert text to speech
def speak(text):
    def speak_async():
        engine.say(text)
        engine.runAndWait()

    # Run the speak function in a separate thread
    threading.Thread(target=speak_async).start()

# 1. File Upload
pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

if pdf_file:
    # Read the uploaded PDF file into memory
    pdf_bytes = BytesIO(pdf_file.read())

    # Extract text from the PDF using pdfplumber
    with pdfplumber.open(pdf_bytes) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

    # Create documents for langchain (using Document class)
    documents = [Document(page_content=text)]

    # 2. Prepare the Chat Prompt Template
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    # 3. Split the Text
    text_splitter = RecursiveCharacterTextSplitter()
    split_docs = text_splitter.split_documents(documents)

    # 4. Create Document Embeddings using Gemini
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY  # Note: using google_api_key parameter
    )
    vector = FAISS.from_documents(split_docs, embeddings)

    # 5. Set up Google Gemini
    gemini = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=GEMINI_API_KEY,  # Note: using google_api_key parameter
        temperature=0
    )

    # 6. Build the Retrieval Chain
    retriever = vector.as_retriever()
    document_chain = create_stuff_documents_chain(gemini, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

     # 7. User Input and Response Generation (via Text or Voice)
    st.subheader("Ask a question based on the PDF content:")

    # Voice Command
    if st.button("Use Voice Command"):
        st.info("Listening... Please speak your question.")

        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
            audio = recognizer.listen(source)

        try:
            # Convert audio to text
            user_query = recognizer.recognize_google(audio)
            st.write(f"Your question: {user_query}")
            
            # Generate response
            response = retrieval_chain.invoke({"input": user_query})
            answer = response['answer']
            st.write(f"Answer: {answer}")
            
            # Convert the answer to speech
            speak(answer)
            
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand your speech.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
    
    # Text Input
    else:
        user_query = st.text_input("Or type your question here:")

        if user_query:
            # Generate response
            response = retrieval_chain.invoke({"input": user_query})
            answer = response['answer']
            st.write(f"Answer: {answer}")
            
            # Convert the answer to speech
            speak(answer)

else:
    st.info("Please upload a PDF document to begin.")
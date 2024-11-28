import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Setting Up the Environment
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# 1. Load the PDF Document
pdf_loader = PyPDFLoader(r"C:\Users\arjun\Documents\Resume - Arjun M S\Arjun_M_S_Aug_2024_Resume.pdf")
docs = pdf_loader.load()

# 2. Prepare the Chat Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# 3. Split the Text
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# 4. Create Document Embeddings using Gemini
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY  # Note: using google_api_key parameter
)
vector = FAISS.from_documents(documents, embeddings)

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

# 7. User Input and Response Generation
response = retrieval_chain.invoke({"input": "what is SQL Ease?"})
print(response["answer"])
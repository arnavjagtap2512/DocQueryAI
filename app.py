from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import streamlit as st
from docx import Document
import time
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Access the API keys
google_api_key = os.getenv('GOOGLE_API_KEY')

# Initialize Streamlit app title and description
st.title("DocQuery AI")
st.write("Instant answers from multiple documents. Transforming document research with AI. Unlock insights effortlessly.")

# File uploader for various file types
uploaded_files = st.file_uploader("Upload your documents", type=["pdf", "docx", "txt", "csv"], accept_multiple_files=True)

# Initialize session state variables
if "processed" not in st.session_state:
    st.session_state["processed"] = False

# Function to extract text from PDF file
def extract_text_from_PDF(pdf_file):
    reader = PdfReader(pdf_file)
    extracted_text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        extracted_text += page.extract_text() + "\n"
    return extracted_text

# Function to extract text from Word document
def extract_text_from_word(docx_file):
    doc = Document(docx_file)
    extracted_text = ""
    for para in doc.paragraphs:
        extracted_text += para.text + "\n"
    return extracted_text

# Function to extract text from text file
def extract_text_from_txt(txt_file):
    extracted_text = txt_file.read().decode('utf-8') + "\n"
    return extracted_text

# Function to extract text from CSV file
def extract_text_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    extracted_text = ""
    for col in df.columns:
        extracted_text += col + " " + " ".join(df[col].astype(str)) + "\n"
    return extracted_text

# Function to chunk text into manageable parts
def chunk_text(text, chunk_size=400):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def stream_data(script):
    for word in script.split(" "):
        yield word + " "
        time.sleep(0.02)

# Process files if uploaded and not already processed
if uploaded_files and not st.session_state["processed"]:
    with st.spinner("Processing Files...."):
        text_from_files = ""
        for file in uploaded_files:
            if file.type == "application/pdf":
                text = extract_text_from_PDF(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = extract_text_from_word(file)
            elif file.type == "text/plain":
                text = extract_text_from_txt(file)
            elif file.type == "text/csv":
                text = extract_text_from_csv(file)
            else:
                st.error(f"Unsupported file type: {file.type}")
                continue

            if text:
                text_from_files += text
            else:
                st.error(f"Failed to extract text from {file.name}")

        file_chunks = chunk_text(text_from_files)

        # Initialize embeddings with Google Generative AI
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Store embeddings in FAISS
        if file_chunks:
            library = FAISS.from_texts(file_chunks, embedding_function)
            st.session_state["library"] = library
            st.session_state["processed"] = True
            st.success("Processed", icon="✅")

# Create a prompt template for user queries
template = """
Question: {question}

Context for answering question: {context}

Instructions: Use the information provided in the context to generate a relevant answer.
"""

# Create a PromptTemplate object
prompt_template = PromptTemplate(input_variables=["question", "context"], template=template)

# Initialize the ChatGoogleGenerativeAI instance
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)

# Display user query input and answer generation button if files are processed
if st.session_state["processed"]:
    user_query = st.text_input(
        "Ask any question about the document",
        placeholder="How many people were given free ration?"
    )
    if st.button("Get Answer"):
        with st.spinner("Generating Answer...."):
            retrieved_chunks = st.session_state["library"].similarity_search(query=user_query, k=5)

            context = ""
            for chunk in retrieved_chunks:
                context += chunk.page_content

            prompt = prompt_template.format(question=user_query, context=context)
            result = llm.invoke(prompt)

            st.write("Question: ", user_query)
            st.write("Answer:")
            st.write_stream(stream_data(result.content))

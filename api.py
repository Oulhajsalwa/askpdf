from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import tempfile
import os
from dotenv import load_dotenv
import uuid

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
from google.generativeai import configure
configure(api_key=api_key)

app = FastAPI()

@app.post("/process_pdf/")
async def process_pdf(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    session_id = str(uuid.uuid4())
    raw_text = ""
    
    try:
        for file in files:
            pdf_content = await file.read()  # Lire le contenu du fichier
            raw_text += get_pdf_text(pdf_content)
        text_chunks = get_text_chunks(raw_text)
        vector_store_path = f"faiss_index_{session_id}"
        get_vector_store(text_chunks, vector_store_path)
        return {"message": "PDF processed successfully", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/ask/")
async def ask_question(session_id: str, question: str):
    if not session_id or not question:
        raise HTTPException(status_code=400, detail="Session ID and question are required")

    vector_store_path = f"faiss_index_{session_id}"
    
    if not os.path.exists(vector_store_path):
        raise HTTPException(status_code=404, detail="Session ID not found")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        new_db = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Error loading FAISS database: {e}")
    
    docs = new_db.similarity_search(question)
    chain = get_conversational_chain()
    
    try:
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        return {"answer": response["output_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

def extract_text_from_images(images):
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

def get_pdf_text(pdf_content: bytes):
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_content)
        temp_pdf_path = temp_pdf.name
    
    pdf_reader = PdfReader(temp_pdf_path)
    for i in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[i]
        page_text = page.extract_text()
        if page_text:
            text += page_text
        else:
            images = convert_from_path(temp_pdf_path, first_page=i+1, last_page=i+1)
            text += extract_text_from_images(images)
    
    os.remove(temp_pdf_path)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, path):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(path)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

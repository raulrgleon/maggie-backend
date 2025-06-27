from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
import os
import openai

# Crear instancia de FastAPI
app = FastAPI()

# Permitir CORS para frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de clave (usa tu OpenAI o OpenRouter)
openai.api_key = os.getenv("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")

# Configuración básica
embedding = OpenAIEmbeddings()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
vectorstore = Chroma(persist_directory="vectores", embedding_function=embedding)

# Modelo de pregunta
class Pregunta(BaseModel):
    pregunta: str

# Extrae texto de PDF
def extraer_texto_pdf(pdf_bytes):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n".join([page.get_text() for page in doc])

# Endpoint para subir y vectorizar PDF
@app.post("/subir-documento")
async def subir_documento(file: UploadFile = File(...)):
    contenido = await file.read()
    texto = extraer_texto_pdf(contenido)
    docs = splitter.create_documents([texto])
    vectorstore.add_documents(docs)
    return { "mensaje": "✅ Documento vectorizado con éxito." }

# Endpoint para preguntar
@app.post("/preguntar")
def responder(p: Pregunta):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(p.pregunta)
    contexto = "\n\n".join([d.page_content for d in docs])

    prompt = f"""Responde de forma legal y precisa según este contexto:\n\n{contexto}\n\nPregunta: {p.pregunta}"""

    respuesta = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            { "role": "system", "content": "Eres un asistente legal muy preciso." },
            { "role": "user", "content": prompt }
        ]
    )

    return { "respuesta": respuesta.choices[0].message.content }

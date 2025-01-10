import os
from fastapi import FastAPI
from dotenv import load_dotenv, find_dotenv
from typing import Union
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

load_dotenv(find_dotenv())

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

app = FastAPI()
model = ChatOpenAI(model="gpt-4o")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get('/chat/test')
def test_chat():
    messages = [
        SystemMessage("Translate the following from English into Italian"),
        HumanMessage("hi!"),
    ]
    
    response = model.invoke(messages)
    return {"response": response}

# Function to load and process PDFs
def load_pdfs_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            documents.extend(loader.load())
    return documents

# Function to create a RAG pipeline with the new structure
def create_rag_pipeline(documents):
    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Create embeddings for the text chunks
    embeddings = OpenAIEmbeddings()

    # Create a vector store for the embeddings
    vector_store = FAISS.from_documents(texts, embeddings)

    # Define a prompt
    prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Define a custom function to format the documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    print("format_docs: ", vector_store.as_retriever() | format_docs)

    # Create the RAG pipeline
    qa_chain = (
        {
            "context": vector_store.as_retriever() | format_docs,
            "input": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )
    return qa_chain

@app.get('/chat/pdfs')
async def chat_with_pdfs():
    # Load and process PDFs
    documents = load_pdfs_from_folder('./pdfs')

    # Create RAG pipeline
    qa_chain = create_rag_pipeline(documents)

    # Run the query through the RAG pipeline
    answer = qa_chain.invoke('Crea KPIs para: Legado Connect es un proyecto emblematico de Legado que se desarrolla en el territorio de origen Guainia- Colombia y funciona como una herramienta habilitadora de beneficios que genera circulos virtuosos de desarrollo sostenible en las comunidades a traves de la conectividad y tecnologia bajo las premisas de la educacion para el desarrollo sostenible, la generacion de contenidos propios y la conservacion de las tradiciones y practicas ancestrales de las comunidades. El objetivo del proyecto es generar beneficios en las comunidades amazonicas entorno a: la gobernanza local, la promocion del interes por la educacion y el aprendizaje, impulsando medidas de prevencion en salud y emergencias, facilitando la preservacion de la cultura, los deportes autoctonos, la biodiversidad, y acercando el entretenimiento a las comunidades, todos estos beneficios habilitados a traves de la conectividad y tecnologia. En lo corrido del ano 2024 Legado Connect cuenta con resultados de impacto en 5 comunidades del Territorio (Concordia, Berrocal, Punta Tigre, Remanso y Zamuro) en donde se establecieron quioscos digitales equipados, logrando mas de 700 personas con acceso 24/7 a conectividad satelital, mas de 80 personas capacitadas en desarrollo de contenido y alfabetizacion digital, en terminos de gobernanza se han generado capacitaciones con enfasis en derechos humanos, liderazgo, igualdad de genero, con la participacion presencial y virtual de mas de 250 personas entre autoridades indigenas, lideres comunitarios, hombres, mujeres y ninos gracias a los puntos de activacion digital en las comunidades mencionadas. En terminos de biodiversidad se cuenta con diagnosticos de flora y fauna e identificacion de corredores biologicos; frente a salud y emergencias se han entregado kits de atencion basica a medicos tradicionales, parteras y personas de respuesta a emergencias. En terminos de contenidos se han desarrollado 4 recorridos digitales 360º de 3 comunidades con alta calidad, y 1 recorrido y star laps a los cerros de Mavicure, respecto a contenidos propios generados como resultado de las capacitaciones se cuenta con mas de 200 registros realizados por las comunidades indigenas. En relacion con deportes ancestrales se apoyo la realizacion de olimpiadas multietnica integrada por los 6 resguardos con participacion de mas de 5 practicas deportivas ancestrales y con un total de 1.200 personas vinculadas.')

    return {"answer": answer}
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DOCS_FOLDER = os.path.join("Files", "Docs")
INDEX_FOLDER = "files_index"

def load_docs():
    documents = []
    for file_name in os.listdir(DOCS_FOLDER):
        path = os.path.join(DOCS_FOLDER, file_name)

        if file_name.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)



        elif file_name.lower().endswith(".docx"):
            loader = Docx2txtLoader(path)



        elif file_name.lower().endswith((".txt", ".md")):
            loader = TextLoader(path, encoding="utf-8")


        else:
            print("Skipped:", file_name)
            continue

        documents.extend(loader.load())
        print("Loaded doc:", file_name)
    return documents

# load_docs()

#CHUNNKING
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vector_db():
    documents = load_docs()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(INDEX_FOLDER)
    print(f"Saved {len(chunks)} chunks to {INDEX_FOLDER}")

build_vector_db()

# Retrieval
vector_db = None
def search(question, k=4):
    global vector_db
    if vector_db is None:
        vector_db = FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)
    return vector_db.similarity_search(question, k=k)
search("python")

# Import the search function from your rag.py file
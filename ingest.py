import os
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# --- Sabitler ---
EMBEDDING_MODEL = "models/text-embedding-004" 
DB_PATH = "rag_store"
DOCS_PATH = "data_docs"

def main():
    print("--- RAG Veritabanı Oluşturma Başladı ---")
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("HATA: GEMINI_API_KEY ortam değişkeni ayarlanmadı.")

    # 1. Dokümanları Yükleme (Sadece TXT uzantılıları)
    loader = DirectoryLoader(
        DOCS_PATH,
        glob="**/*.txt",
        loader_kwargs={'encoding': 'utf-8', 'errors': 'ignore'}
    )
    docs = loader.load()

    # 2. Parçalara Ayırma (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)

    if not chunks:
        raise SystemExit("HATA: Dokümanlar bulunamadı veya boş.")

    print(f"Toplam parça (chunk) sayısı: {len(chunks)}")

    # 3. Embedding ve Vektör Veritabanı (API Anahtarı ile)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=os.getenv("GEMINI_API_KEY"))

    # ChromaDB'yi oluşturma/kaydetme
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    vector_store.persist()

    print(f"✅ BAŞARILI: {len(chunks)} parça eklendi.")

if __name__ == "__main__":
    main()

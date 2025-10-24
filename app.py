import streamlit as st
import os
from pathlib import Path
from typing import List

# LangChain ve Gemini Kütüphaneleri (Streamlit ile uyumlu importlar)
from langchain_core.prompts import PromptTemplate 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA 
from langchain_community.vectorstores import Chroma 
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Sabitler ve Ayarlar ---
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004" 
DB_PATH = "rag_store"
DOCS_PATH = "data_docs" 

# API Anahtarını sadece ortam değişkeninden çeker (Streamlit'in Secrets hatasını atlamak için)
API_KEY = os.getenv("GEMINI_API_KEY") 

# --- RAG Zinciri Kurulumu ---
@st.cache_resource
def load_rag_chain():
    """RAG zincirini, LLM'i yükler ve veritabanını kontrol/oluşturur."""

    api_key = API_KEY

    if not api_key:
        st.error("❌ HATA: GEMINI_API_KEY bulunamadı. Lütfen Terminal'de export komutuyla ayarlayın.")
        return None, None

    # 1. Embedding Fonksiyonu
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)

    # 2. Veritabanının varlığını kontrol et ve oluştur (Yalnızca yoksa)
    if not Path(DB_PATH).exists():
        try:
            # Dokümanları yükleme
            loader = DirectoryLoader(
                DOCS_PATH, glob="**/*.txt", loader_kwargs={'encoding': 'utf-8', 'errors': 'ignore'}
            )
            docs = loader.load()

            # Parçalara ayırma (Chunking)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)

            if not chunks:
                st.error("HATA: Otomatik indeksleme başarısız oldu. Dokümanlar boş veya okunamıyor.")
                return None, None

            vector_store = Chroma.from_documents(
                documents=chunks, embedding=embeddings, persist_directory=DB_PATH
            )
            st.success("✅ Veritabanı ilk çalıştırmada başarıyla oluşturuldu!")

        except Exception as e:
            st.error(f"FATAL HATA: Otomatik indeksleme sırasında beklenmeyen hata oluştu: {e}")
            return None, None

    else:
        # Veritabanı varsa, sadece yükle
        vector_store = Chroma(
            persist_directory=DB_PATH, embedding_function=embeddings
        )

    # 3. RAG Zincirini Kurma
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, google_api_key=api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    prompt_template = """
    Sen bir Biyomedikal Bilgi Asistanısın. Aşağıdaki biyomedikal metinleri (Context) kullanarak, kullanıcıya Türkçe ve net bir şekilde yanıt ver. 
    Yanıtların teknik, kısa ve direkt olmalıdır. Bağlamda bulamadığın sorulara 'Bu konuda elimde yeterli bilgi yok.' diye yanıt ver.

    Context: {context}
    Soru: {question}
    Yanıt:
    """
    PROMPT = PromptTemplate.from_template(prompt_template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain, retriever 

# RAG zincirini bir kez başlat
QA_CHAIN, RETRIEVER = load_rag_chain()


# --- Streamlit Ana Uygulaması ---
def main():
    st.set_page_config(page_title="Biyomedikal RAG Asistanı", layout="wide")

    # Başlık ve Etik Uyarı
    st.markdown(
        """
        <div style="text-align: center; background-color: #1F618D; padding: 15px; border-radius: 10px; color: white;">            <h1>🔬 Biyomedikal RAG Bilgi Asistanı</h1>
            <p>Gemini AI, LangChain ve ChromaDB ile güçlendirilmiştir.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.warning("🚨 ETİK UYARI: Bu sistem tıbbi tanı, tedavi veya kişisel sağlık tavsiyesi VERMEZ. Sadece bilgi asistanıdır.")

    # RAG sistemini başlat
    if not QA_CHAIN:
        st.stop() # Hata varsa durdur

    # Chat Mesajları
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Dokümanlarımdaki konularla (İmmünoloji, Etik, Cihazlar) ilgili sorular sorun."}]

    # Arayüzü iki sütuna ayırma
    col1, col2 = st.columns([3, 1])

    with col1: # Ana Sohbet Sütunu
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Kullanıcıdan Girdi Alma
        if prompt := st.chat_input("Biyomedikal sorunuzu buraya yazın..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("🧠 Gemini yanıt oluşturuyor..."):
                # RAG zincirini çalıştırma
                try:
                    result = QA_CHAIN({"query": prompt})
                    response = result['result']
                    docs = result['source_documents']

                    # Kaynakları ve cevabı birleştir
                    sources_list = "\n".join([f"- **{d.metadata.get('source', 'Bilinmeyen')}**" for d in docs])

                    full_response = response + "\n\n**Çekilen Kaynaklar:**\n" + sources_list

                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    with st.chat_message("assistant"):
                        st.markdown(full_response)

                except Exception as e:
                    st.error(f"Yanıt oluşturulamadı. Hata: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "Üzgünüm, bir sorun oluştu."})

    with col2: # Kaynak ve Detay Sütunu
        st.subheader("İpuçları ve Kaynaklar")
        st.info("Bu model, sadece veri setinde bulunan 14 adet biyomedikal dokümandan bilgi çeker.")
        st.markdown("**Örnek Sorular:**")
        st.markdown("- Kalbin en güçlü odacığı nedir?")
        st.markdown("- Tıbbi cihazların sınıflandırılması nasıl yapılır?")
        st.markdown("- Genetik mühendisliğinde CRISPR nedir?")
        st.markdown("- Aksiyon potansiyelini başlatan temel fiziksel mekanizma nedir?")
        st.markdown("- Biyomedikal araştırmalarda etik kurallardan biri olan Özerklik ne anlama gelir?")


if __name__ == "__main__":
    main()
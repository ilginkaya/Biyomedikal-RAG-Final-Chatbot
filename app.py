import streamlit as st
import os
from pathlib import Path

# --- LangChain 0.2+/0.3+ uyumlu importlar (helpers kullanmadan) ---
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # paket adı bu

# --- Sabitler ve Ayarlar ---
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
DB_PATH = "rag_store"
DOCS_PATH = "data_docs"

API_KEY = os.getenv("GEMINI_API_KEY")


@st.cache_resource
def load_rag_chain():
    """RAG zinciri, LLM ve vektör veritabanını hazırlar."""
    if not API_KEY:
        st.error("❌ HATA: GEMINI_API_KEY bulunamadı. Lütfen ortam değişkeni olarak ayarlayın.")
        return None, None

    # 1) Embedding
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=API_KEY)

    # 2) Chroma DB oluştur/yükle
    if not Path(DB_PATH).exists():
        try:
            loader = DirectoryLoader(
                DOCS_PATH, glob="**/*.txt", loader_kwargs={"encoding": "utf-8", "errors": "ignore"}
            )
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            if not chunks:
                st.error("HATA: İndeksleme başarısız. Dokümanlar boş veya okunamıyor.")
                return None, None

            vector_store = Chroma.from_documents(
                documents=chunks, embedding=embeddings, persist_directory=DB_PATH
            )
        except Exception as e:
            st.error(f"FATAL HATA: Otomatik indeksleme sırasında hata: {e}")
            return None, None
    else:
        vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # 3) LLM ve Retriever
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, google_api_key=API_KEY)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 4) Prompt (create_stuff_documents_chain => {context} + {input})
    prompt_template = """
Sen bir Biyomedikal Bilgi Asistanısın. Aşağıdaki biyomedikal metinleri (Context) kullanarak, kullanıcıya Türkçe ve net bir şekilde yanıt ver.
Yanıtların teknik, kısa ve direkt olmalıdır. Bağlamda bulamadığın sorulara 'Bu konuda elimde yeterli bilgi yok.' diye yanıt ver.

Context:
{context}

Soru:
{input}

Yanıt:
"""
    PROMPT = PromptTemplate.from_template(prompt_template)

    # 5) Dokümanları LLM'e "stuff" eden zincir
    doc_chain = create_stuff_documents_chain(llm=llm, prompt=PROMPT)

    # 6) create_retrieval_chain yerine RunnableParallel + pipe
    #    - retriever girdiyi alır ve {context} üretir
    #    - RunnablePassthrough aynı girdiyi {input} olarak geçirir
    qa_chain = RunnableParallel(
        {"context": retriever, "input": RunnablePassthrough()}
    ) | doc_chain

    return qa_chain, retriever


# Zinciri başlat
QA_CHAIN, RETRIEVER = load_rag_chain()


def main():
    st.set_page_config(page_title="Biyomedikal RAG Asistanı", layout="wide")

    st.markdown(
        """
        <div style="text-align: center; background-color: #1F618D; padding: 15px; border-radius: 10px; color: white;">
            <h1>🔬 Biyomedikal RAG Bilgi Asistanı</h1>
            <p>Gemini AI, LangChain ve ChromaDB ile güçlendirilmiştir.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.warning("🚨 ETİK UYARI: Bu sistem tıbbi tanı, tedavi veya kişisel sağlık tavsiyesi VERMEZ. Sadece bilgi asistanıdır.")

    if not QA_CHAIN:
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Merhaba! Dokümanlarımdaki konularla (İmmünoloji, Etik, Cihazlar) ilgili sorular sorun."
        }]

    col1, col2 = st.columns([3, 1])

    with col1:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if prompt := st.chat_input("Biyomedikal sorunuzu buraya yazın..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("🧠 Gemini yanıt oluşturuyor..."):
                try:
                    # Zinciri çağır
                    result = QA_CHAIN.invoke(prompt)
                    # create_stuff_documents_chain bazı sürümlerde "answer", bazılarında "output_text" döndürür
                    response = result.get("answer") or result.get("output_text") or ""
                    # Kaynakları ayrıca al (pipeline çıktısında yok)
                    docs = RETRIEVER.get_relevant_documents(prompt)
                    sources_list = "\n".join([f"- **{d.metadata.get('source', 'Bilinmeyen')}**" for d in docs])

                    full_response = response + ("\n\n**Çekilen Kaynaklar:**\n" + sources_list if sources_list else "")
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    with st.chat_message("assistant"):
                        st.markdown(full_response)

                except Exception as e:
                    st.error(f"Yanıt oluşturulamadı. Hata: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "Üzgünüm, bir sorun oluştu."})

    with col2:
        st.subheader("İpuçları ve Kaynaklar")
        st.info("Bu model, sadece sizin yüklediğiniz biyomedikal dokümanlardan bilgi çeker.")
        st.markdown("**Örnek Sorular:**")
        st.markdown("- Kalbin en güçlü odacığı nedir?")
        st.markdown("- Tıbbi cihazların sınıflandırılması nasıl yapılır?")
        st.markdown("- Genetik mühendisliğinde CRISPR nedir?")


if __name__ == "__main__":
    main()
import streamlit as st
import os
from pathlib import Path

# LangChain 0.2+/0.3+ uyumlu importlar
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Ayarlar ---
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
DB_PATH = "rag_store"
DOCS_PATH = "data_docs"


def get_api_key():
    """API anahtarını ENV > Secrets sırasıyla al."""
    return os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")


def _join_docs(docs):
    """Retriever'dan gelen dokümanları tek metne çevirir."""
    return "\n\n".join(d.page_content for d in docs)


@st.cache_resource
def load_rag_chain():
    api_key = get_api_key()
    if not api_key:
        st.error("❌ HATA: GEMINI_API_KEY bulunamadı. Lütfen Secrets veya ortam değişkeni olarak ekleyin.")
        return None, None

    # 1) Embedding
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)

    # 2) Chroma DB (oluştur/yükle)
    if not Path(DB_PATH).exists():
        try:
            if not Path(DOCS_PATH).exists():
                st.error(f"HATA: '{DOCS_PATH}' klasörü yok. Metin dosyalarını buraya koymalısın.")
                return None, None

            loader = DirectoryLoader(
                DOCS_PATH, glob="**/*.txt", loader_kwargs={"encoding": "utf-8", "errors": "ignore"}
            )
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)

            if not chunks:
                st.error("HATA: İndeksleme başarısız. Dokümanlar boş veya okunamıyor.")
                return None, None

            vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
        except Exception as e:
            st.error(f"FATAL HATA: Otomatik indeksleme sırasında hata: {e}")
            return None, None
    else:
        vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # 3) LLM ve Retriever
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, google_api_key=api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 4) Prompt (context + input)
    template = """
Sen bir Biyomedikal Bilgi Asistanısın. Aşağıdaki biyomedikal metinleri (Context) kullanarak, kullanıcıya Türkçe ve net bir şekilde yanıt ver.
Yanıtların teknik, kısa ve direkt olmalıdır. Bağlamda bulamadığın sorulara 'Bu konuda elimde yeterli bilgi yok.' diye yanıt ver.

Context:
{context}

Soru:
{input}

Yanıt:
"""
    PROMPT = PromptTemplate.from_template(template)

    # 5) LCEL zinciri:
    # retriever -> dokümanları getir -> tek metne çevir
    context_chain = retriever | RunnableLambda(_join_docs)

    # {context, input} -> prompt -> llm -> düz metin
    qa_chain = (
        RunnableParallel({"context": context_chain, "input": RunnablePassthrough()})
        | PROMPT
        | llm
        | StrOutputParser()
    )

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

    # Bilgilendirici küçük etiket (anahtar nereden geliyor)
    src = "ENV" if os.getenv("GEMINI_API_KEY") else ("SECRETS" if "GEMINI_API_KEY" in st.secrets else "YOK")
    st.caption(f"API anahtarı kaynağı: {src}")

    if not QA_CHAIN:
        st.stop()

    # Sohbet durumu
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

        prompt = st.chat_input("Biyomedikal sorunuzu buraya yazın...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("🧠 Gemini yanıt oluşturuyor..."):
                try:
                    answer_text = QA_CHAIN.invoke(prompt)

                    # Kaynak listesi (pipeline çıktısında yok; retriever'dan ayrıca çekiyoruz)
                    docs = RETRIEVER.get_relevant_documents(prompt)
                    sources_list = "\n".join([f"- **{d.metadata.get('source', 'Bilinmeyen')}**" for d in docs])

                    full_response = answer_text + ("\n\n**Çekilen Kaynaklar:**\n" + sources_list if sources_list else "")
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
        st.markdown("- Kalbin en güçlü odacığı nedir? Sebebini kısa açıklar mısın?")
        st.markdown("- Tıbbi cihazların sınıflandırılması nasıl yapılır?")
        st.markdown("- Genetik mühendisliğinde CRISPR nedir?")
        st.markdown("- MRG'nin çalışma prensibini teknik ve kısa açıklar mısın?")
        st.markdown("- Biyomalzemelerin vücutta gösterdiği üç farklı biyo-davranış şekli nedir?")

if __name__ == "__main__":
    main()
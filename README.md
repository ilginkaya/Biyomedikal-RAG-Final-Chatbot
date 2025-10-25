# 🔬 Biyomedikal RAG Bilgi Asistanı (Gemini, LangChain & Streamlit)

## 📌 1. Projenin Amacı

Bu proje, **Akbank GenAI Bootcamp** kapsamında geliştirilmiştir.  
Amacı, **Retrieval-Augmented Generation (RAG)** temelli bir chatbot geliştirerek, kullanıcıların biyomedikal alanındaki teknik metinlerden kaynak göstererek bilgi edinmesini sağlamaktır.

**Çözülen Problem:**  
Biyomedikal mühendisliği alanında çok sayıda karmaşık ve yapılandırılmamış bilgi bulunur. Bu proje, PDF/TXT/MD formatındaki içeriklerden **doğru, hızlı ve kaynaklı bilgi çekimi** sağlayarak bu sorunu çözer.

**Kullanılan Teknolojiler:**  
Gemini LLM, LangChain, ChromaDB, Streamlit

---

## 📚 2. Veri Seti Hazırlama

- **Konu:**  
  Biyomedikal mühendisliği temelleri, immünoloji, tıbbi görüntüleme, biyoetik ve cihaz regülasyonları gibi **14 farklı alt konu**.
  
- **Hazırlık Metodolojisi:**  
  Akademik kaynaklardan alınan bilgiler düzenlenerek, proje amaçlarına uygun biçimde **14 adet TXT/MD dosyası** olarak yapılandırılmıştır (`data_docs/` klasöründe).

---

## ⚙️ 3. Çözüm Mimarisi ve Yöntemler

| Bileşen | Teknoloji | Amaç |
| :--- | :--- | :--- |
| **LLM (Language Model)** | Gemini-2.5-flash | Nihai yanıtı üretir. |
| **Embedding Model** | `models/text-embedding-004` | Metinleri vektör uzayına dönüştürür. |
| **Vektör Veritabanı** | ChromaDB | Belgeleri saklar ve benzerlik sorgularını yönetir. |
| **RAG Framework** | LangChain | Retrieval (bilgi çekimi) ve Generation (yanıt üretimi) süreçlerini birleştirir. |
| **Arayüz** | Streamlit | Web tabanlı kullanıcı etkileşimini sağlar. |

**RAG Süreci:**  
Kullanıcı bir soru yazar → ChromaDB’den en alakalı metin parçaları çekilir → Bu bilgiler Gemini modeline bağlam olarak verilir → Model, bağlama dayalı teknik bir yanıt üretir.

---

## 🧩 4. Kurulum ve Çalıştırma Kılavuzu

### Gereksinimler
- Python 3.10+
- GEMINI API Anahtarı
- `requirements.txt` dosyasındaki bağımlılıklar

### Kurulum Adımları
1.  **Reposu Klonlama:**
    ```bash
    git clone https://github.com/ilginkaya/Biyomedikal-RAG-Final-Chatbot.git
    cd Biyomedikal-RAG-Final-Chatbot
    ```
2.  **Sanal Ortam Kurulumu:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Bağımlılıkların Kurulumu:**
    ```bash
    pip3 install -r requirements.txt
    ```
4.  **API Anahtarı Tanımlama:**
    ```bash
    export GEMINI_API_KEY="[Kendi API Anahtarınız]"
    ```
5.  **Uygulamayı Çalıştırma:**
    ```bash
    python3 app.py
    ```
Local URL'niz ile sayfaya ulaşabilrsiniz.

## 5. Web Arayüzü & Product Kılavuzu 

**Elde Edilen Sonuçlar Özeti:**

* **Başarı:** Proje, Gemini API, LangChain ve ChromaDB entegrasyonunu başarıyla göstererek RAG mimarisini hayata geçirmiştir.
* **Kabiliyet:** Chatbot, sadece yüklenen biyomedikal metinlerden bilgi çekerek doğru ve konuya özgü yanıtlar üretmektedir.

### Çalışma Akışı ve Görsel Kılavuz
Kullanıcı, arayüzde sorusunu sorar. Chatbot, otomatik olarak oluşturulan veritabanından bilgi çeker. Cevabın altında, bilginin hangi kaynaktan (hangi TXT/MD dosyasından) alındığı gösterilir.

**Ekran Görüntüsü (Çalışma Örneği):**

<img width="1470" height="686" alt="Ekran Resmi 2025-10-25 12 29 26" src="https://github.com/user-attachments/assets/b1b05f38-a8f1-4ed0-8628-79465a2db5c7" />


### Test Senaryosu Örnekleri
| Soru | İlgili Alan | Beklenen Yanıt Tipi |
| :--- | :--- | :--- |
| "Aksiyon potansiyelini başlatan temel fiziksel mekanizma nedir?" | Biyofizik | İyonların hücre zarı boyunca hareketini açıklayan yanıt. |
| "Biyomedikal araştırmalarda etik kurallardan biri olan Özerklik ne anlama gelir?" | Biyoetik | Hastanın karar verme hakkını açıklayan yanıt. |
| "Fransa'nın başkenti neresidir?" | RAG Sınırlandırma Testi | "Bu konuda elimde yeterli bilgi yok." (RAG izolasyonunun kanıtı). |

***

### 🔗 Deploy Linki
[Canlı Uygulama Linki](https://biyomedikal-rag-final-chatbot-gckekrqhzbdrsug3ri8pxj.streamlit.app/)



import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import google.generativeai as genai

# === Configure Gemini API Key (Streamlit Secrets or Env Var) ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAJoievCdhnH4VUJjTVZ-Vkp1J3v1D53ao")
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found. Please set it in Streamlit secrets or environment variables.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)


# === Streamlit UI ===
st.set_page_config(page_title="📄 Chat with your PDF", layout="wide")
st.title("📄 Chat with your PDF using Gemini 2.5 Pro")

uploaded_file = st.file_uploader("📤 Upload your PDF", type="pdf")
user_question = st.text_input("💬 Ask a question about the PDF")


# === Gemini Embedding Function ===
class GeminiEmbeddings(Embeddings):
    """A wrapper for the Gemini embedding model that conforms to LangChain's Embeddings interface."""

    def embed_documents(self, texts):
        """Embeds a list of documents."""
        return [
            genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )["embedding"] for text in texts
        ]

    def embed_query(self, text):
        """Embeds a single query."""
        return genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )["embedding"]


# === Vector Store Builder (Caching Enabled) ===
@st.cache_resource(show_spinner="⚡ Building vector store...")
def build_vector_store(_chunks, _embedding_function):
    """Builds and caches a FAISS vector store from document chunks."""
    vectorstore = FAISS.from_documents(
        documents=_chunks,
        embedding=_embedding_function
    )
    return vectorstore


# === Main Execution ===
if uploaded_file and user_question:
    with st.spinner("⚙️ Processing PDF and finding answer..."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Load and split PDF
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load_and_split()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(pages)

            # Create embedding class
            gemini_embeddings = GeminiEmbeddings()

            # Build vector store (cached)
            vectorstore = build_vector_store(chunks, gemini_embeddings)

            # Get top matching chunks
            docs = vectorstore.similarity_search(user_question, k=4)

            # Prepare context + prompt
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"""
You are a helpful assistant specialized in answering questions based strictly on the provided context.

Context from the PDF:
---
{context}
---

Question: {user_question}

Answer:
"""

            # Run Gemini Model
            model = genai.GenerativeModel("gemini-2.5-pro")
            response = model.generate_content(prompt)

            # === Safely Extract Answer ===
            answer = ""
            if response and response.candidates:
                for candidate in response.candidates:
                    if candidate.content.parts:
                        answer = "".join(
                            [part.text for part in candidate.content.parts if hasattr(part, "text")]
                        )
                        break  # take first valid candidate

            # === Display Answer ===
            st.markdown("### 📘 Answer")
            st.write(answer if answer else "⚠️ No answer generated. Try rephrasing your question or check API quota.")

            # === Optional: Show Retrieved Chunks ===
            with st.expander("📄 Show retrieved context chunks"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**Chunk {i+1} (Source: Page {doc.metadata.get('page', 'N/A')})**")
                    st.info(doc.page_content)

            # === Debug Raw Response ===
            with st.expander("🛠 Debug Raw Gemini Response"):
                st.json(response.to_dict())

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
        finally:
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)


import os
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from werkzeug.utils import secure_filename
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import cohere

# Load environment variables
load_dotenv()

# Configure Cohere client
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

# Configure Gemini client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Flask app
app = Flask(__name__)

# Chat history (in memory)
History = []

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ------------------------------------------------------------------ Upload Route ------------------------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload_data():
    try:
        raw_text = request.form.get("text")
        file = request.files.get("file")

        raw_docs = []

        if file and file.filename != "":
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            loader = PyPDFLoader(filepath)
            raw_docs = loader.load()

        elif raw_text and raw_text.strip():
            raw_docs = [Document(page_content=raw_text)]

        else:
            return jsonify({"error": "No file or text provided"}), 400

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        chunked_docs = text_splitter.split_documents(raw_docs)

        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        # Upload to Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME")
        index = pc.Index(index_name)

        vectors = []
        for i, doc in enumerate(chunked_docs):
            vector = embeddings.embed_query(doc.page_content)
            vectors.append({
                "id": str(i),
                "values": vector,
                "metadata": {"text": doc.page_content}
            })

        index.upsert(vectors=vectors)

        return jsonify({"message": "✅ Data uploaded and stored in Pinecone!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------ Query Rewriting ------------------------------------------------------------------

def transform_query(question: str) -> tuple[str, float]:
    """Rewrite a follow-up question into a standalone query."""
    History.append({"role": "user", "parts": [{"text": question}]})

    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction="""You are a query rewriting expert.
        Based on the provided chat history, rephrase the "Follow Up user Question"
        into a complete, standalone question that can be understood without the chat history.
        Only output the rewritten question and nothing else."""
    )

    t0 = time.time()
    response = model.generate_content(History)
    elapsed = time.time() - t0

    History.pop()  # Remove so it doesn't pollute future turns

    return response.text.strip(), elapsed


# ------------------------------------------------------------------ Retrieval + Reranking ------------------------------------------------------------------

def get_context(query: str) -> str:
    """Retrieve and rerank top documents from Pinecone."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    query_vector = embeddings.embed_query(query)

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    pinecone_index = pc.Index(index_name)

    # Retrieve from Pinecone
    search_results = pinecone_index.query(
        vector=query_vector,
        top_k=10,
        include_metadata=True
    )

    documents = [match["metadata"]["text"] for match in search_results["matches"]]

    if not documents:
        return ""

    # Rerank with Cohere
    rerank_results = cohere_client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=5
    )

    reranked_docs = [documents[r.index] for r in rerank_results.results]
    context = "\n\n---\n\n".join(reranked_docs)

    return context


# ------------------------------------------------------------------ Generation ------------------------------------------------------------------

def chat_with_gemini(query: str, context: str) -> tuple[str, float, int]:
    """Generate an answer using Gemini with retrieved context."""
    History.append({"role": "user", "parts": [{"text": query}]})

    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=f"""You will be given a context of relevant information and a user question.
        Your task is to answer the user's question based ONLY on the provided context.
        If the answer is not in the context, say:
        "I could not find the answer in the provided document."
        Keep your answers clear, concise, and educational.

        Context: {context}
        """
    )

    t0 = time.time()
    response = model.generate_content(History)
    elapsed = time.time() - t0

    History.append({"role": "model", "parts": [{"text": response.text}]})

    tokens_used = len(query.split()) + len(context.split())
    return response.text, elapsed, tokens_used


# ------------------------------------------------------------------ Flask Routes ------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json(force=True)
    question = data.get("question") or data.get("query") or ""

    if not question:
        return jsonify({"error": "No question provided"}), 400

    rewritten_query, rewrite_time = transform_query(question)
    context = get_context(rewritten_query)

    if not context:
        return jsonify({
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": "I could not find the answer in the provided document.",
            "sources": [],
            "runtime": round(rewrite_time, 2),
            "tokens": 0
        })

    answer, gen_time, tokens = chat_with_gemini(rewritten_query, context)
    sources = context.split("\n\n---\n\n")

    return jsonify({
        "question": question,
        "rewritten_query": rewritten_query,
        "answer": answer,
        "sources": sources[1:3],
        "runtime": round(rewrite_time + gen_time, 2),
        "tokens": tokens
    })


# ------------------------------------------------------------------ Entry Point ------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

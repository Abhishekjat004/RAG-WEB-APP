import os
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from werkzeug.utils import secure_filename
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import cohere



start = time.time()
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

# -------------------------------------------------------------File or text Uploader --------------------------------------------------

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_data():
    try:
        raw_text = request.form.get("text")  # text sent via form-data
        file = request.files.get("file")     # file sent via form-data

        raw_docs = []

        # Case 1: PDF upload
        if file and file.filename != "":
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            loader = PyPDFLoader(filepath)
            raw_docs = loader.load()

        # Case 2: Raw text upload
        elif raw_text and raw_text.strip():
            raw_docs = [Document(page_content=raw_text)]

        else:
            return jsonify({"error": "No file or text provided"}), 400

        # Step 2: Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        chunked_docs = text_splitter.split_documents(raw_docs)

        # Step 3: Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        # Step 4: Upload to Pinecone
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






# ---------------------------------------------------------Transfrom Query -------------------------------------------------------
def transform_query(question: str) -> str:
    """Rewrite follow-up question into standalone query"""
    History.append({"role": "user", "parts": [{"text": question}]})

    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction="""You are a query rewriting expert.
        Based on the provided chat history, rephrase the "Follow Up user Question"
        into a complete, standalone question that can be understood without the chat history.
        Only output the rewritten question and nothing else."""
    )

    response = model.generate_content(History)
    History.pop()           # Remove the transform_query to avoid confusion in future interactions

    elapsed = time.time() - start
    return response.text, elapsed

# ------------------------------------------------------ get context from pinecone vector database----------------------------------------------------------

def get_context(query: str) -> str:
    """Retrieve top documents from Pinecone as context"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    query_vector = embeddings.embed_query(query)

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    pinecone_index = pc.Index(index_name)

    # Step 1: Retrieve from Pinecone
    search_results = pinecone_index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True
    )

    documents = [match['metadata']['text'] for match in search_results['matches']]

    # Step 2: Rerank with Cohere
    rerank_results = cohere_client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=5  # final top docs
    )

    # Step 3: Build context string
    reranked_docs = [documents[r.index] for r in rerank_results.results]
    context = "\n\n---\n\n".join(reranked_docs)

    return context


# ----------------------------------------------------------chat_with_gemini ----------------------------------------------------------
def chat_with_gemini(query: str, context: str) -> str:
    """Generate answer using Gemini with context"""
    History.append({"role": "user", "parts": [{"text": query}]})

    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=f"""You will be given a context of relevant information and a user question.
        Your task is to answer the user's question based ONLY on the provided context.
        If the answer is not in the context, you must say:
        "I could not find the answer in the provided document."
        Keep your answers clear, concise, and educational.

        Context: {context}
        """
    )

    response = model.generate_content(History)

    History.append({"role": "model", "parts": [{"text": response.text}]})
    elapsed = time.time() - start

    tokens_used = len(query.split()) + len(context.split())
    return response.text, context, elapsed, tokens_used



# ----------------------------------------------------------------Flask Routes ------------------------------------------------------------
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
    answer, context, runtime, tokens = chat_with_gemini(rewritten_query, context)
    sources = context.split("\n\n---\n\n")
    return jsonify({
        "question": question,
        "rewritten_query": rewritten_query,
        "answer": answer,
        "sources" : sources[1:3],
        "runtime": round(runtime + rewrite_time, 2),
        "tokens": tokens
    })



# ---------- Run Flask ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render gives a PORT env var
    app.run(host="0.0.0.0", port=port)  








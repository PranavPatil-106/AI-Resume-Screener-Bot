import os
from dotenv import load_dotenv
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

load_dotenv()

class HuggingFaceVectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):

        """
        Initialize the vector store with FAISS and sentence transformers.
        
        Args:
            model_name: Sentence transformer model name
        """

        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.model_name = model_name
        self.use_hf_api = False
        if self.hf_token:
            # Prefer using HF Inference API when token available (avoids local model download)
            self.use_hf_api = True
            self.model = None
        else:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                print(f"Warning: Could not load {model_name}, falling back to a simpler approach")
                # Fallback to a basic embedding approach
                self.model = None
        self.index = None
        self.vectors = []
        self.documents = []
        self.dimension = None

    def embed_text(self, text: str):
        """
        Generate embeddings for the given text.
        Uses HF Inference API if HF token is available; otherwise uses local sentence-transformers.
        """
        if self.use_hf_api:
            import requests
            headers = {"Authorization": f"Bearer {self.hf_token}", "Accept": "application/json"}
            url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
            try:
                # API check
                resp = requests.post(url, headers=headers, json={"inputs": text, "options": {"wait_for_model": True}}, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                # Handle various return formats
                arr = np.array(data, dtype=np.float32)
                if arr.ndim == 2:
                    emb = arr.mean(axis=0)
                elif arr.ndim == 3:
                    emb = arr.mean(axis=(0, 1))
                else:
                    emb = arr.astype(np.float32)
                return emb
            except Exception as e:
                print(f"Warning: HF Inference API embedding failed: {e}.")
                # If API fails, we should NOT fall back to random hash. 
                # Ideally we should fallback to local model if possible, or raise error.
                if self.model is None:
                     # Try to load local model as fallback if not already loaded
                    try:
                        print("Attempting to load local fallback model...")
                        self.model = SentenceTransformer(self.model_name)
                    except Exception as local_e:
                        raise RuntimeError(f"Both API and local model failed. API Error: {e}. Local Error: {local_e}")

        if self.model is not None:
            return self.model.encode([text])[0]
        else:
            raise RuntimeError("No embedding model available (API failed or not configured, and local model failed).")

    def embed_batch(self, texts: list):
        """
        Generate embeddings for a batch of texts.
        """
        if self.use_hf_api:
            # HF API often accepts list of strings, but check specific model/task support. 
            # For safety, we loop or use model-specific batching. 
            # Replaced simply loop here to reuse the robust embed_text logic or implement specific batch call.
            # To differ to robustness, we will loop for now unless we are sure about the endpoint limits.
            return [self.embed_text(t) for t in texts]
        
        if self.model is not None:
            return self.model.encode(texts)
        
        raise RuntimeError("No embedding model available.")

    def add_documents(self, docs):
        """
        Add documents to the vector store by generating embeddings.
        Args:
            docs: List of documents with 'resume_id' and 'content' keys
        """
        if not docs:
            return

        texts = [doc["content"] for doc in docs]
        
        # improved batch embedding
        try:
            embeddings_list = self.embed_batch(texts)
        except Exception:
            # Fallback to single processing if batch fails
            embeddings_list = [self.embed_text(t) for t in texts]

        # Collect all documents and generate embeddings
        new_embeddings = []
        for doc, embedding in zip(docs, embeddings_list):
            self.documents.append(doc["content"])
            new_embeddings.append(embedding)
            self.vectors.append({
                "resume_id": doc["resume_id"],
                "embedding": embedding,
                "content": doc["content"]
            })
        
        # Initialize FAISS index if not already done
        if new_embeddings:
            new_embeddings_array = np.array(new_embeddings).astype('float32')
            faiss.normalize_L2(new_embeddings_array)
            
            if self.index is None:
                self.dimension = new_embeddings_array.shape[1]
                self.index = faiss.IndexFlatIP(self.dimension)
                self.index.add(new_embeddings_array)
            else:
                self.index.add(new_embeddings_array)

    def similarity_search(self, query, top_k=3):
        """
        Perform similarity search to find the most relevant documents.
        Args:
            query: Search query text
            top_k: Number of top results to return
        Returns:
            List of top_k most similar documents
        """
        if not self.vectors or self.index is None:
            return []
        
        query_emb = self.embed_text(query)
        query_emb = query_emb.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_emb)
        
        # Search using FAISS
        # Ensure we don't ask for more neighbors than we have vectors
        k = min(top_k, len(self.vectors))
        scores, indices = self.index.search(query_emb, k)
        
        results = []
        if scores.size > 0 and indices.size > 0:
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.vectors) and idx >= 0:
                     results.append(self.vectors[idx])
        
        return results

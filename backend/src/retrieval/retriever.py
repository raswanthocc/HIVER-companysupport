"""
Modern Multi-Brand RAG Retriever using ChromaDB and SentenceTransformers.
Supports brand-filtered retrieval across TWCS dataset.
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from src.models import RetrievedContext

class ResolutionRetriever:
    """
    Retrieves historical resolved interactions using a local Vector Database (ChromaDB).
    Allows filtering by brand handle (e.g., 'AmazonHelp', 'CompanySupport', 'SpotifyCares').
    """
    def __init__(self, db_path: str = "data/chroma_db", retrieval_corpus_path: Optional[str] = None, kb_path: Optional[str] = None, **kwargs):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if not os.path.isabs(db_path):
            db_path = os.path.join(base_dir, db_path)

        self.db_path = db_path
        self.retrieval_corpus_path = retrieval_corpus_path
        self.kb_path = kb_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self._init_client()

    def _init_client(self):
        if os.path.exists(self.db_path):
            try:
                self.client = chromadb.PersistentClient(path=self.db_path)
                self.collection = self.client.get_collection(name="twcs_support")
            except Exception:
                self.client = None
                self.collection = None
        else:
            self.client = None
            self.collection = None

    def get_available_brands(self) -> List[str]:
        """Return list of distinct brands indexed in ChromaDB."""
        if not hasattr(self, 'collection') or self.collection is None:
            self._init_client()
            
        if self.collection is None:
            return ["All Brands"]
            
        try:
            # Query sample of metadatas to get unique brands
            sample = self.collection.get(limit=2000, include=["metadatas"])
            brands = set()
            if sample and sample.get('metadatas'):
                for m in sample['metadatas']:
                    if m and 'brand' in m:
                        brands.add(m['brand'])
            sorted_brands = sorted(list(brands))
            return ["All Brands"] + sorted_brands
        except Exception:
            return ["All Brands"]

    def retrieve(self, query: str, brand: str = "All Brands", top_k: int = 3) -> RetrievedContext:
        """
        Retrieve matching historical pairs for the given query, optionally filtered by brand.
        """
        if not hasattr(self, 'collection') or self.collection is None:
            self._init_client()

        if self.collection is None:
            return RetrievedContext(kb_guide={}, historical_exemplars=[], similarity_scores=[], grounding_status="no_db_found")

        query_embedding = self.model.encode([query]).tolist()
        
        # Apply metadata filter if a specific brand is selected
        where_filter = None
        if brand and brand != "All Brands":
            where_filter = {"brand": brand}

        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                where=where_filter
            )
        except Exception as e:
            # Fallback if where query fails or empty
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )

        exemplars = []
        scores = []
        
        if results and results.get('metadatas') and len(results['metadatas'][0]) > 0:
            for idx in range(len(results['metadatas'][0])):
                meta = results['metadatas'][0][idx]
                dist = results['distances'][0][idx] if 'distances' in results and results['distances'] else 0.0
                doc = results['documents'][0][idx] if 'documents' in results and results['documents'] else ""
                
                item = {
                    "customer_text": doc,
                    "agent_text": meta.get("agent_reply", ""),
                    "brand": meta.get("brand", "Unknown"),
                    "similarity_score": round(float(dist), 4)
                }
                exemplars.append(item)
                scores.append(round(float(dist), 4))
                
        return RetrievedContext(
            kb_guide={},
            historical_exemplars=exemplars,
            similarity_scores=scores,
            grounding_status="vector_db_retrieval"
        )

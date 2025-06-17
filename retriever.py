import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Represents a document in the knowledge base"""
    id: str
    title: str
    text: str
    metadata: Optional[Dict] = None

class BaseRetriever(ABC):
    """Abstract base class for retrievers"""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        pass
    
    @abstractmethod
    def index_documents(self, documents: List[Document]):
        pass

class BM25Retriever(BaseRetriever):
    """BM25-based sparse retriever"""
    
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.stop_words = set(stopwords.words('english'))
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Tokenize and preprocess text"""
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        return tokens
    
    def index_documents(self, documents: List[Document]):
        """Index documents for BM25 retrieval"""
        self.documents = documents
        corpus = []
        
        for doc in documents:
            # Combine title and text for better retrieval
            full_text = f"{doc.title} {doc.text}"
            tokens = self._preprocess_text(full_text)
            corpus.append(tokens)
        
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"Indexed {len(documents)} documents with BM25")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve documents using BM25"""
        if self.bm25 is None:
            raise ValueError("Documents must be indexed before retrieval")
        
        query_tokens = self._preprocess_text(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k document indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return top documents with scores
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            # Add score to metadata
            doc_copy = Document(
                id=doc.id,
                title=doc.title,
                text=doc.text,
                metadata={**(doc.metadata or {}), 'bm25_score': scores[idx]}
            )
            results.append(doc_copy)
        
        return results

class DenseRetriever(BaseRetriever):
    """Dense retriever using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.index = None
        self.embeddings = None
    
    def index_documents(self, documents: List[Document]):
        """Index documents using dense embeddings"""
        self.documents = documents
        
        # Create text for embedding (title + text)
        texts = []
        for doc in documents:
            full_text = f"{doc.title} {doc.text}"
            texts.append(full_text)
        
        # Generate embeddings
        logger.info("Generating embeddings for documents...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        logger.info(f"Indexed {len(documents)} documents with dense embeddings")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve documents using dense similarity"""
        if self.index is None:
            raise ValueError("Documents must be indexed before retrieval")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Return top documents
        results = []
        for score, idx in zip(scores[0], indices[0]):
            doc = self.documents[idx]
            doc_copy = Document(
                id=doc.id,
                title=doc.title,
                text=doc.text,
                metadata={**(doc.metadata or {}), 'dense_score': float(score)}
            )
            results.append(doc_copy)
        
        return results

class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining BM25 and dense retrieval"""
    
    def __init__(self, dense_model: str = "all-MiniLM-L6-v2", bm25_weight: float = 0.5):
        self.bm25_retriever = BM25Retriever()
        self.dense_retriever = DenseRetriever(dense_model)
        self.bm25_weight = bm25_weight
        self.dense_weight = 1.0 - bm25_weight
    
    def index_documents(self, documents: List[Document]):
        """Index documents with both retrievers"""
        self.bm25_retriever.index_documents(documents)
        self.dense_retriever.index_documents(documents)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve using hybrid approach"""
        # Get results from both retrievers
        bm25_results = self.bm25_retriever.retrieve(query, top_k * 2)
        dense_results = self.dense_retriever.retrieve(query, top_k * 2)
        
        # Combine scores
        doc_scores = {}
        
        # Add BM25 scores
        for doc in bm25_results:
            doc_id = doc.id
            bm25_score = doc.metadata.get('bm25_score', 0)
            doc_scores[doc_id] = {'doc': doc, 'bm25': bm25_score, 'dense': 0}
        
        # Add dense scores
        for doc in dense_results:
            doc_id = doc.id
            dense_score = doc.metadata.get('dense_score', 0)
            if doc_id in doc_scores:
                doc_scores[doc_id]['dense'] = dense_score
            else:
                doc_scores[doc_id] = {'doc': doc, 'bm25': 0, 'dense': dense_score}
        
        # Normalize and combine scores
        bm25_scores = [info['bm25'] for info in doc_scores.values()]
        dense_scores = [info['dense'] for info in doc_scores.values()]
        
        if bm25_scores:
            bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1
            bm25_scores = [s / bm25_max for s in bm25_scores]
        
        if dense_scores:
            dense_max = max(dense_scores) if max(dense_scores) > 0 else 1
            dense_scores = [s / dense_max for s in dense_scores]
        
        # Calculate final scores
        final_scores = []
        for i, (doc_id, info) in enumerate(doc_scores.items()):
            final_score = (self.bm25_weight * bm25_scores[i] + 
                          self.dense_weight * dense_scores[i])
            final_scores.append((final_score, doc_id, info['doc']))
        
        # Sort by final score and return top-k
        final_scores.sort(reverse=True)
        results = []
        
        for score, doc_id, doc in final_scores[:top_k]:
            doc_copy = Document(
                id=doc.id,
                title=doc.title,
                text=doc.text,
                metadata={**(doc.metadata or {}), 'hybrid_score': score}
            )
            results.append(doc_copy)
        
        return results

class WikipediaKnowledgeBase:
    """Wikipedia-based knowledge base for DeepRAG"""
    
    def __init__(self, retriever_type: str = "hybrid"):
        self.retriever_type = retriever_type
        self.retriever = self._create_retriever(retriever_type)
        self.documents = []
    
    def _create_retriever(self, retriever_type: str) -> BaseRetriever:
        """Factory method to create retriever"""
        if retriever_type == "bm25":
            return BM25Retriever()
        elif retriever_type == "dense":
            return DenseRetriever()
        elif retriever_type == "hybrid":
            return HybridRetriever()
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
    
    def load_documents_from_file(self, file_path: str):
        """Load documents from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.documents = []
        for item in data:
            doc = Document(
                id=item.get('id', str(len(self.documents))),
                title=item.get('title', ''),
                text=item.get('text', ''),
                metadata=item.get('metadata', {})
            )
            self.documents.append(doc)
        
        self.retriever.index_documents(self.documents)
        logger.info(f"Loaded {len(self.documents)} documents")
    
    def create_sample_knowledge_base(self):
        """Create a sample knowledge base for demonstration"""
        sample_docs = [
            Document(
                id="1",
                title="Peter's Friends",
                text="Peter's Friends is a 1992 British comedy-drama film written and directed by Kenneth Branagh. The film stars Kenneth Branagh, Stephen Fry, Hugh Laurie, Emma Thompson, Imelda Staunton, and Rita Rudner."
            ),
            Document(
                id="2", 
                title="Kenneth Branagh",
                text="Sir Kenneth Charles Branagh is a British actor and filmmaker. Branagh was born on 10 December 1960 in Belfast, Northern Ireland, to Frances and William Branagh."
            ),
            Document(
                id="3",
                title="The Lord of the Rings: The Fellowship of the Ring",
                text="The Lord of the Rings: The Fellowship of the Ring is a 2001 epic fantasy adventure film directed by Peter Jackson. The film has a runtime of 178 minutes in its theatrical version."
            ),
            Document(
                id="4",
                title="The Lord of the Rings: The Two Towers", 
                text="The Lord of the Rings: The Two Towers is a 2002 epic fantasy adventure film directed by Peter Jackson. The film has a runtime of 179 minutes in its theatrical version."
            ),
            Document(
                id="5",
                title="The Lord of the Rings: The Return of the King",
                text="The Lord of the Rings: The Return of the King is a 2003 epic fantasy drama film directed by Peter Jackson. The film has a runtime of 201 minutes in its theatrical version."
            ),
            Document(
                id="6",
                title="Belfast",
                text="Belfast is the capital and largest city of Northern Ireland, standing on the banks of the River Lagan on the east coast. Kenneth Branagh was born in Belfast in 1960."
            )
        ]
        
        self.documents = sample_docs
        self.retriever.index_documents(sample_docs)
        logger.info("Created sample knowledge base with 6 documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve documents for a query"""
        return self.retriever.retrieve(query, top_k)
    
    def format_retrieved_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents as context"""
        if not docs:
            return "No relevant documents found."
        
        context = "Retrieved information:\n"
        for i, doc in enumerate(docs, 1):
            context += f"{i}. {doc.title}: {doc.text}\n"
        
        return context

# Example usage
if __name__ == "__main__":
    # Create knowledge base
    kb = WikipediaKnowledgeBase("hybrid")
    kb.create_sample_knowledge_base()
    
    # Test retrieval
    query = "Who directed Peter's Friends?"
    results = kb.retrieve(query, top_k=3)
    
    print(f"Query: {query}")
    print(f"Retrieved {len(results)} documents:")
    for doc in results:
        print(f"- {doc.title}: {doc.text[:100]}...")
        if doc.metadata:
            print(f"  Score: {doc.metadata}") 
"""
Clean semantic search implementation using ChromaDB and Sentence Transformers
"""
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

class CustomEmbeddingFunction():
    """Custom embedding function for ChromaDB using sentence transformers"""
    
    NAME = "thenlper/gte-small"

    def __init__(self, model_name: str = "thenlper/gte-small"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.name = model_name  # Chroma expects this attribute

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Convert texts to embeddings."""
        return self.model.encode(texts).tolist()


class SemanticSearchDB:
    """Main semantic search database class"""
    
    def __init__(self, db_path: str = "./search_db"):
        """Initialize semantic search database."""
        self.db_path = db_path

        # Use Chroma's sentence transformers directly
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name='thenlper/gte-small'
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = "documents"

        # Check if collection already exists
        existing_collections = [col.name for col in self.client.list_collections()]
        if self.collection_name in existing_collections:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Loaded existing collection with {self.collection.count()} documents")
        else:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Document search collection"}
            )
            print("Created new collection")

    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the database"""
        texts = []
        metadatas = []
        ids = []

        for i, doc in enumerate(documents):
            content = doc.get('content', '').strip()
            if len(content) < 20:  # Skip very short content
                continue

            # Split long content into chunks
            chunks = self._split_into_chunks(content, max_size=1024, overlap=250)

            for j, chunk in enumerate(chunks):
                texts.append(chunk)
                metadata = {
                    'title': doc.get('title', ''),
                    'url': doc.get('url', ''),
                    'username': doc.get('username', ''),
                    'category': self._categorize_content(chunk, doc.get('title', '')),
                    'source_doc_id': i,
                    'chunk_id': j,
                    'total_chunks': len(chunks),
                    'timestamp': datetime.now().isoformat()
                }
                metadatas.append(metadata)
                ids.append(f"doc_{i}_chunk_{j}")

        # Batch insertion to respect ChromaDB's limit
        max_batch_size = 5000
        total = len(texts)
        for i in range(0, total, max_batch_size):
            batch_texts = texts[i:i + max_batch_size]
            batch_ids = ids[i:i + max_batch_size]
            batch_metadatas = metadatas[i:i + max_batch_size]
            print(f"Adding batch {i} to {i + len(batch_texts)}")
            self.collection.add(
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

        print(f"✅ Added {len(texts)} document chunks to database")

    
    def _split_into_chunks(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for punct in ['. ', '! ', '? ']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct > start:
                        end = last_punct + len(punct)
                        break
                else:
                    # Fall back to word boundary
                    last_space = text.rfind(' ', start, end)
                    if last_space > start:
                        end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = max(start + 1, end - overlap)
            
            if start >= len(text):
                break
        
        return chunks
    
    def _categorize_content(self, content: str, title: str = "") -> str:
        """Categorize content based on keywords"""
        text = (content + " " + title).lower()
        
        categories = {
            'assignment': ['assignment', 'homework', 'ga1', 'ga2', 'ga3','ga4', 'ga5', 'ga6','ga7', 'project', 'project 1', 'project 2'],
            'exam': ['exam', 'test', 'final', 'roe'],
            'technical': ['code', 'error', 'python', 'api', 'debug'],
            'course': ['course', 'syllabus', 'schedule', 'deadline'],
            'general': []  # default category
        }
        
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return 'general'
    
    def semantic_search(self, query: str, n_results: int = 5, 
                       category_filter: Optional[str] = None) -> Dict[str, Any]:
        """Perform semantic search using embeddings"""
        where_clause = {}
        if category_filter:
            where_clause["category"] = category_filter
        
        try:
            if where_clause:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=min(n_results, self.collection.count()),
                    where=where_clause
                )
            else:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=min(n_results, self.collection.count())
                )
            
            return {
                'documents': results['documents'][0] if results['documents'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                'distances': results['distances'][0] if results['distances'] else []
            }
        except Exception as e:
            print(f"Semantic search error: {e}")
            return {'documents': [], 'metadatas': [], 'distances': []}
    
    def keyword_search(self, query: str, n_results: int = 5,
                      category_filter: Optional[str] = None) -> Dict[str, Any]:
        """Perform keyword-based search"""
        # Get all documents
        try:
            if category_filter:
                all_docs = self.collection.get(where={"category": category_filter})
            else:
                all_docs = self.collection.get()
            
            documents = all_docs.get('documents', [])
            metadatas = all_docs.get('metadatas', [])
            ids = all_docs.get('ids', [])
            
            # Score documents based on keyword matches
            query_words = query.lower().split()
            scored_docs = []
            
            for i, doc in enumerate(documents):
                doc_lower = doc.lower()
                score = sum(doc_lower.count(word) for word in query_words)
                
                if score > 0:
                    scored_docs.append({
                        'document': doc,
                        'metadata': metadatas[i],
                        'id': ids[i],
                        'score': score
                    })
            
            # Sort by score and return top results
            scored_docs.sort(key=lambda x: x['score'], reverse=True)
            top_docs = scored_docs[:n_results]
            
            return {
                'documents': [doc['document'] for doc in top_docs],
                'metadatas': [doc['metadata'] for doc in top_docs],
                'scores': [doc['score'] for doc in top_docs]
            }
            
        except Exception as e:
            print(f"Keyword search error: {e}")
            return {'documents': [], 'metadatas': [], 'scores': []}
    
    def hybrid_search(self, query: str, n_results: int = 5,
                     category_filter: Optional[str] = None,
                     semantic_weight: float = 0.7) -> Dict[str, Any]:
        """Combine semantic and keyword search results"""
        
        # Get semantic results
        semantic_results = self.semantic_search(
            query, n_results=n_results*2, category_filter=category_filter
        )
        
        # Get keyword results
        keyword_results = self.keyword_search(
            query, n_results=n_results*2, category_filter=category_filter
        )
        
        # Combine and score results
        combined_results = {}
        
        # Add semantic results
        for i, (doc, meta, dist) in enumerate(zip(
            semantic_results['documents'],
            semantic_results['metadatas'],
            semantic_results['distances']
        )):
            doc_id = meta.get('source_doc_id', f"sem_{i}")
            semantic_score = 1.0 - min(dist, 1.0)  # Convert distance to similarity
            
            combined_results[doc_id] = {
                'document': doc,
                'metadata': meta,
                'semantic_score': semantic_score,
                'keyword_score': 0.0,
                'final_score': semantic_score * semantic_weight
            }
        
        # Add keyword results
        for i, (doc, meta, score) in enumerate(zip(
            keyword_results['documents'],
            keyword_results['metadatas'],
            keyword_results['scores']
        )):
            doc_id = meta.get('source_doc_id', f"key_{i}")
            keyword_score = min(score / 10.0, 1.0)  # Normalize keyword score
            
            if doc_id in combined_results:
                combined_results[doc_id]['keyword_score'] = keyword_score
                combined_results[doc_id]['final_score'] = (
                    combined_results[doc_id]['semantic_score'] * semantic_weight +
                    keyword_score * (1.0 - semantic_weight)
                )
            else:
                combined_results[doc_id] = {
                    'document': doc,
                    'metadata': meta,
                    'semantic_score': 0.0,
                    'keyword_score': keyword_score,
                    'final_score': keyword_score * (1.0 - semantic_weight)
                }
        
        # Sort by final score and return top results
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )[:n_results]
        
        return {
            'documents': [r['document'] for r in sorted_results],
            'metadatas': [r['metadata'] for r in sorted_results],
            'scores': [r['final_score'] for r in sorted_results],
            'semantic_scores': [r['semantic_score'] for r in sorted_results],
            'keyword_scores': [r['keyword_score'] for r in sorted_results]
        }
    
    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Simple search method for compatibility"""
        return self.hybrid_search(query, n_results=n_results)

    def evaluate_search(self, test_queries: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate search performance using MAP, Precision, and Recall
        
        test_queries format:
        [
            {
                'query': 'search query',
                'relevant_doc_ids': ['doc_1', 'doc_2', ...],
                'category': 'optional_category_filter'
            }
        ]
        """
        all_avg_precisions = []
        all_precisions = []
        all_recalls = []
        
        for test_case in test_queries:
            query = test_case['query']
            relevant_ids = set(test_case['relevant_doc_ids'])
            category = test_case.get('category')
            
            # Perform search
            results = self.hybrid_search(query, n_results=10, category_filter=category)
            
            # Get retrieved document IDs
            retrieved_ids = []
            for meta in results['metadatas']:
                doc_id = f"doc_{meta.get('source_doc_id', '')}"
                retrieved_ids.append(doc_id)
            
            # Calculate metrics
            precision, recall, avg_precision = self._calculate_metrics(
                retrieved_ids, relevant_ids
            )
            
            all_avg_precisions.append(avg_precision)
            all_precisions.append(precision)
            all_recalls.append(recall)
        
        return {
            'MAP': np.mean(all_avg_precisions) if all_avg_precisions else 0.0,
            'Precision': np.mean(all_precisions) if all_precisions else 0.0,
            'Recall': np.mean(all_recalls) if all_recalls else 0.0
        }
    
    def _calculate_metrics(self, retrieved_ids: List[str], 
                          relevant_ids: set) -> Tuple[float, float, float]:
        """Calculate precision, recall, and average precision"""
        if not retrieved_ids or not relevant_ids:
            return 0.0, 0.0, 0.0
        
        # Precision and Recall
        retrieved_set = set(retrieved_ids)
        relevant_retrieved = retrieved_set.intersection(relevant_ids)
        
        precision = len(relevant_retrieved) / len(retrieved_set) if retrieved_set else 0.0
        recall = len(relevant_retrieved) / len(relevant_ids) if relevant_ids else 0.0
        
        # Average Precision
        avg_precision = 0.0
        relevant_count = 0
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                avg_precision += precision_at_i
        
        avg_precision = avg_precision / len(relevant_ids) if relevant_ids else 0.0
        
        return precision, recall, avg_precision
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            count = self.collection.count()
            all_data = self.collection.get()
            metadatas = all_data.get('metadatas', [])
            
            # Count by category
            categories = {}
            for meta in metadatas:
                category = meta.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
            
            return {
                'total_documents': count,
                'categories': categories,
                'embedding_model': "thenlper/gte-small"
            }
        except Exception as e:
            return {'error': str(e)}
    
    def search_by_category(self, category: str, limit: int = 10) -> Dict[str, Any]:
        """Get all documents from a specific category"""
        try:
            results = self.collection.get(
                where={"category": category},
                limit=limit
            )
            return {
                'documents': results.get('documents', []),
                'metadatas': results.get('metadatas', [])
            }
        except Exception as e:
            print(f"Error getting category content: {e}")
            return {'documents': [], 'metadatas': []}
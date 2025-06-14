import chromadb
from chromadb.config import Settings
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os
from bs4 import BeautifulSoup

class TDSKnowledgeBase:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="tds_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def add_course_content(self, content: str, source: str = "course_content"):
        """Add course content to the knowledge base"""
        chunks = self.chunk_text(content, chunk_size=500)
        
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                'source': source,
                'type': 'course_content',
                'chunk_id': i
            })
            ids.append(f"course_{source}_{i}")
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metadata to ChromaDB-compatible format"""
        sanitized = {}
        
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = None
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                if value:  # Non-empty list
                    sanitized[key] = ', '.join(str(item) for item in value)
                else:  # Empty list
                    sanitized[key] = ''
            else:
                # Convert other types to string
                sanitized[key] = str(value)
        
        return sanitized
    
    def add_document(self, text: str, doc_id: str, metadata: Dict[str, Any]):
        """Add a single document to the knowledge base"""
        # Chunk long documents
        chunks = self.chunk_text(text, chunk_size=800, overlap=100)
        
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Skip very short chunks
                continue
                
            documents.append(chunk)
            
            # Create metadata for this chunk
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_id'] = i
            chunk_metadata['total_chunks'] = len(chunks)
            
            # Sanitize metadata for ChromaDB
            sanitized_metadata = self.sanitize_metadata(chunk_metadata)
            
            metadatas.append(sanitized_metadata)
            ids.append(f"{doc_id}_chunk_{i}")
        
        if documents:  # Only add if we have valid chunks
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
    
    def clean_html(self, html_content: str) -> str:
        """Clean HTML content to plain text"""
        if not html_content:
            return ""
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text().strip()
    
    def add_discourse_posts(self, posts_file: str = "data/tds_discourse_posts.json"):
        """Add Discourse posts to the knowledge base - Updated for new scraper format"""
        if not os.path.exists(posts_file):
            print(f"❌ File {posts_file} not found")
            return False
            
        with open(posts_file, 'r', encoding='utf-8') as f:
            posts = json.load(f)
        
        print(f"📚 Processing {len(posts)} topics from discourse...")
        
        topics_processed = 0
        posts_processed = 0
        
        for topic in posts:
            try:
                topic_id = topic.get('id')
                topic_title = topic.get('title', 'No title')
                topic_url = topic.get('url', '')
                
                # Process the main topic (first post)
                topic_posts = topic.get('posts', [])
                if not topic_posts:
                    continue
                
                # Main topic post
                main_post = topic_posts[0]
                main_content = main_post.get('text', '')
                
                if len(main_content.strip()) > 50:
                    topic_text = f"TOPIC: {topic_title}\n\n{main_content}"
                    
                    # Add image information if available
                    topic_images = topic.get('images', [])
                    if topic_images:
                        topic_text += f"\n\nImages in topic ({len(topic_images)}):\n"
                        for img in topic_images:
                            img_desc = img.get('alt', 'Screenshot/Image')
                            topic_text += f"- {img_desc}\n"
                    
                    # Convert tags list to string for metadata
                    tags = topic.get('tags', [])
                    tags_str = ', '.join(tags) if tags else ''
                    
                    self.add_document(
                        text=topic_text,
                        doc_id=f"topic_{topic_id}",
                        metadata={
                            'source': 'discourse',
                            'type': 'topic',
                            'topic_id': topic_id,
                            'title': topic_title,
                            'url': topic_url,
                            'username': main_post.get('username', 'Unknown'),
                            'created_at': topic.get('created_at', ''),
                            'category_id': topic.get('category_id'),
                            'tags': tags_str,  # Convert list to string
                            'image_count': len(topic_images),
                            'reply_count': len(topic_posts) - 1
                        }
                    )
                    posts_processed += 1
                
                # Process replies
                for i, reply_post in enumerate(topic_posts[1:], 1):
                    reply_content = reply_post.get('text', '')
                    
                    if len(reply_content.strip()) > 50:
                        reply_text = f"REPLY TO: {topic_title}\nBy: {reply_post.get('username', 'Unknown')}\n\n{reply_content}"
                        
                        # Add reply-specific images
                        reply_images = reply_post.get('images', [])
                        if reply_images:
                            reply_text += f"\n\nImages in reply ({len(reply_images)}):\n"
                            for img in reply_images:
                                img_desc = img.get('alt', 'Screenshot/Image')
                                reply_text += f"- {img_desc}\n"
                        
                        # Convert tags list to string for metadata
                        tags = topic.get('tags', [])
                        tags_str = ', '.join(tags) if tags else ''
                        
                        self.add_document(
                            text=reply_text,
                            doc_id=f"reply_{topic_id}_{reply_post.get('id', i)}",
                            metadata={
                                'source': 'discourse',
                                'type': 'reply',
                                'topic_id': topic_id,
                                'post_id': reply_post.get('id'),
                                'title': f"Reply to: {topic_title}",
                                'url': topic_url,
                                'username': reply_post.get('username', 'Unknown'),
                                'created_at': reply_post.get('created_at', ''),
                                'category_id': topic.get('category_id'),
                                'tags': tags_str,  # Convert list to string
                                'image_count': len(reply_images),
                                'reply_number': i
                            }
                        )
                        posts_processed += 1
                
                topics_processed += 1
                
                if topics_processed % 10 == 0:
                    print(f"  Processed {topics_processed} topics, {posts_processed} posts...")
                    
            except Exception as e:
                print(f"❌ Error processing topic {topic.get('id', 'unknown')}: {e}")
                continue
        
        print(f"✅ Successfully processed {topics_processed} topics and {posts_processed} posts")
        return True
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
            
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            # Break if we've covered all words
            if i + chunk_size >= len(words):
                break
                
        return chunks
    
    def search(self, query: str, n_results: int = 5, filter_type: str = None) -> Dict[str, Any]:
        """Search for relevant content with optional filtering"""
        try:
            where_clause = None
            if filter_type:
                where_clause = {"type": filter_type}
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause
            )
            
            return {
                'documents': results['documents'][0],
                'metadatas': results['metadatas'][0],
                'distances': results['distances'][0] if 'distances' in results else []
            }
        except Exception as e:
            print(f"Error searching: {e}")
            return {'documents': [], 'metadatas': [], 'distances': []}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        try:
            total_count = self.collection.count()
            
            # Get sample of metadata to analyze types
            sample_results = self.collection.get(limit=min(100, total_count))
            
            type_counts = {}
            source_counts = {}
            
            for metadata in sample_results['metadatas']:
                doc_type = metadata.get('type', 'unknown')
                source = metadata.get('source', 'unknown')
                
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                source_counts[source] = source_counts.get(source, 0) + 1
            
            return {
                'total_documents': total_count,
                'types': type_counts,
                'sources': source_counts
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {'total_documents': 0, 'types': {}, 'sources': {}}
    
    def search_by_user(self, username: str, n_results: int = 10) -> Dict[str, Any]:
        """Search for posts by a specific user"""
        try:
            results = self.collection.query(
                query_texts=[f"posts by {username}"],
                n_results=n_results,
                where={"username": username}
            )
            
            return {
                'documents': results['documents'][0],
                'metadatas': results['metadatas'][0],
                'distances': results['distances'][0] if 'distances' in results else []
            }
        except Exception as e:
            print(f"Error searching by user: {e}")
            return {'documents': [], 'metadatas': [], 'distances': []}
    
    def search_with_images(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for posts that contain images"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"image_count": {"$gt": 0}}
            )
            
            return {
                'documents': results['documents'][0],
                'metadatas': results['metadatas'][0],
                'distances': results['distances'][0] if 'distances' in results else []
            }
        except Exception as e:
            print(f"Error searching with images: {e}")
            return {'documents': [], 'metadatas': [], 'distances': []}
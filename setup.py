"""
Setup script to load data and test the semantic search system
"""

import os
import json
from typing import List, Dict, Any
from VectorDB import SemanticSearchDB


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file (one JSON object per line)"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Invalid JSON on line {line_num}: {e}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return data


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load regular JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def clean_document_data(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean and standardize document data"""
    cleaned = []
    
    for i, item in enumerate(raw_data):
        try:
            # Handle different input formats
            if 'full_text' in item:
                # Format from discourse/forum data
                doc = {
                    'content': item.get('full_text', '').strip(),
                    'title': item.get('title', f'Document {i}'),
                    'url': item.get('url', ''),
                    'username': item.get('username', 'unknown')
                }
            elif 'content' in item:
                # Standard content format
                doc = {
                    'content': item.get('content', '').strip(),
                    'title': item.get('title', f'Document {i}'),
                    'url': item.get('url', ''),
                    'username': item.get('username', 'unknown')
                }
            else:
                print(f"Skipping item {i}: no content field found")
                continue
            
            # Only keep documents with substantial content
            if len(doc['content']) >= 30:
                cleaned.append(doc)
            else:
                print(f"Skipping item {i}: content too short")
                
        except Exception as e:
            print(f"Error processing item {i}: {e}")
    
    return cleaned


def test_search_system(db: SemanticSearchDB) -> None:
    """Test the search system with sample queries"""
    
    test_queries = [
        "assignment deadlines",
        "project requirements", 
        "course evaluation",
        "technical problems",
        "final exam"
    ]
    
    print("\n" + "="*50)
    print("TESTING SEARCH SYSTEM")
    print("="*50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        # Test semantic search
        print("📊 Semantic Search:")
        sem_results = db.semantic_search(query, n_results=3)
        for i, (doc, meta) in enumerate(zip(sem_results['documents'], sem_results['metadatas'])):
            preview = doc[:100] + "..." if len(doc) > 100 else doc
            category = meta.get('category', 'unknown')
            print(f"  {i+1}. [{category}] {preview}")
        
        # Test hybrid search
        print("\n🔀 Hybrid Search:")
        hybrid_results = db.hybrid_search(query, n_results=3)
        for i, (doc, meta, score) in enumerate(zip(
            hybrid_results['documents'], 
            hybrid_results['metadatas'],
            hybrid_results['scores']
        )):
            preview = doc[:100] + "..." if len(doc) > 100 else doc
            category = meta.get('category', 'unknown')
            print(f"  {i+1}. [{category}] Score: {score:.3f} - {preview}")


def test_evaluation_metrics(db: SemanticSearchDB) -> None:
    """Test evaluation metrics with sample data"""
    
    print("\n" + "="*50)
    print("TESTING EVALUATION METRICS")
    print("="*50)
    
    # Sample test cases (in real usage, you'd have ground truth data)
    test_cases = [
        {
            'query': 'assignment deadline',
            'relevant_doc_ids': ['doc_0', 'doc_1'],  # These would be actual relevant doc IDs
            'category': 'assignment'
        },
        {
            'query': 'technical error',
            'relevant_doc_ids': ['doc_2', 'doc_3'],
            'category': 'technical'
        }
    ]
    
    # Note: This is just a demo - you'd need real ground truth data
    print("Note: Using sample test cases for demonstration")
    print("In practice, you'd need actual ground truth relevance judgments")
    
    try:
        metrics = db.evaluate_search(test_cases)
        print(f"\nEvaluation Results:")
        print(f"MAP (Mean Average Precision): {metrics['MAP']:.3f}")
        print(f"Precision: {metrics['Precision']:.3f}")
        print(f"Recall: {metrics['Recall']:.3f}")
    except Exception as e:
        print(f"Error running evaluation: {e}")


def main():
    """Main setup function"""
    
    print("🚀 Setting up Semantic Search System")
    print("=" * 50)
    
    # Initialize database
    db = SemanticSearchDB()
    
    # Look for data files
    data_files = {
        'jsonl': ['file1.jsonl'],
        'json': ['tds_discourse_posts_old.json']
    }
    
    total_loaded = 0
    
    # Load JSONL files
    for filename in data_files['jsonl']:
        if os.path.exists(filename):
            print(f"\n📁 Loading {filename}...")
            raw_data = load_jsonl_file(filename)
            print(f"   Found {len(raw_data)} raw entries")
            
            cleaned_data = clean_document_data(raw_data)
            print(f"   Cleaned: {len(cleaned_data)} valid entries")
            
            if cleaned_data:
                db.add_documents(cleaned_data)
                total_loaded += len(cleaned_data)
        else:
            print(f"📝 {filename} not found")
    
    # Load JSON files
    for filename in data_files['json']:
        if os.path.exists(filename):
            print(f"\n📁 Loading {filename}...")
            raw_data = load_json_file(filename)
            print(f"   Found {len(raw_data)} raw entries")
            
            cleaned_data = clean_document_data(raw_data)
            print(f"   Cleaned: {len(cleaned_data)} valid entries")
            
            if cleaned_data:
                db.add_documents(cleaned_data)
                total_loaded += len(cleaned_data)
        else:
            print(f"📝 {filename} not found")
    
    print(f"\n✅ Total documents loaded: {total_loaded}")
    
    # Show database statistics
    stats = db.get_stats()
    print(f"\n📊 Database Statistics:")
    print(f"   Total chunks: {stats.get('total_documents', 0)}")
    print(f"   Embedding model: {stats.get('embedding_model', 'unknown')}")
    print(f"   Categories: {stats.get('categories', {})}")
    
    # Test the system if we have data
    if total_loaded > 0:
        test_search_system(db)
        test_evaluation_metrics(db)
    else:
        print("\n⚠️  No data loaded - please add some JSON/JSONL files to test")
    
    print(f"\n🎉 Setup complete!")
    print(f"💡 Features available:")
    print(f"   ✅ Semantic search with sentence transformers")
    print(f"   ✅ Keyword search")
    print(f"   ✅ Hybrid search (semantic + keyword)")
    print(f"   ✅ Category filtering")
    print(f"   ✅ Evaluation metrics (MAP, Precision, Recall)")
    print(f"   ✅ Chunk size: 1024 with 250 overlap")


if __name__ == "__main__":
    main()
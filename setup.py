import os
import json
from vector_db import TDSKnowledgeBase

def main():
    print("Setting up TDS Virtual TA...")
    
    # Initialize knowledge base
    kb = TDSKnowledgeBase()
    
    # Add course content
    course_content = """
    Tools in Data Science - Jan 2025
    Tools in Data Science is a practical diploma level data science course at IIT Madras that teaches popular tools for sourcing data, transforming it, analyzing it, communicating these as visual stories, and deploying them in production.

    This course exposes you to real-life tools
    This course is quite hard
    Programming skills are a pre-requisite
    We encourage learning by sharing
    We cover 7 modules in 12 weeks
    The content evolves with technology and feedback. Track the commit history for changes.

    Released content:
    - Development Tools and concepts to build models and apps
    - Deployment Tools and concepts to publish what you built
    - Large Language Models that make your work easier and your apps smarter
    - Data Sourcing to get data from the web, files, and databases
    - Data Preparation to clean up and convert the inputs to the right format
    - Project 1 to build an LLM-based automation agent

    Evaluations are mostly open Internet
    - GA: Graded assignments Best 4 out of 7 (15%)
    - P1: Project 1 Take-home open-Internet (20%)
    - P2: Project 2 Take-home open-Internet (20%)
    - ROE: Remote Online Exam Online open-Internet MCQ (20%)
    - F: Final end-term In-person, no internet, mandatory (25%)

    Important notes:
    - Graded Assignment 1 checks course pre-requisites
    - Remote exams are open and hard - you can use Internet, WhatsApp, ChatGPT, notes, friends
    - Final exam is in-person and closed book, tests memory, and is easy
    - Projects test application in real-world context
    - Use gpt-3.5-turbo-0125 model when specified, even if AI proxy only supports gpt-4o-mini
    - For Docker vs Podman: Podman is recommended for the course, but Docker is acceptable
    
    Course instructor: S. Anand (@s.anand)
    Teaching Assistants: Carlton (@Carlton), Jivraj (@Jivraj)
    """
    
    kb.add_course_content(course_content, "course_overview")
    
    # Look for discourse posts in data directory
    data_file_path = os.path.join("data", "tds_discourse_posts.json")
    
    if os.path.exists(data_file_path):
        print(f"📚 Loading Discourse posts from {data_file_path}...")
        
        try:
            with open(data_file_path, 'r', encoding='utf-8') as f:
                discourse_data = json.load(f)
            
            print(f"Found {len(discourse_data)} posts in the data file")
            
            # Add each post to the knowledge base
            posts_added = 0
            for post in discourse_data:
                try:
                    # Create a comprehensive text representation of the post
                    post_text = f"""
Title: {post.get('title', 'No title')}
Created: {post.get('created_at', 'Unknown date')}
URL: {post.get('url', 'No URL')}
Category ID: {post.get('category_id', 'Unknown')}
Tags: {', '.join(post.get('tags', []))}

Full Content:
{post.get('full_text', 'No content')}

Individual Posts:
"""
                    
                    # Add individual post details
                    for i, individual_post in enumerate(post.get('posts', [])):
                        post_text += f"""
Post {i+1} by {individual_post.get('username', 'Unknown user')} at {individual_post.get('created_at', 'Unknown time')}:
{individual_post.get('text', 'No text')}
---
"""
                    
                    # Add image information if available
                    if post.get('images'):
                        post_text += "\nImages in this post:\n"
                        for img in post['images']:
                            post_text += f"- {img.get('alt', 'No description')}: {img.get('url', 'No URL')}\n"
                    
                    # Add to knowledge base with metadata
                    metadata = {
                        "source": "discourse",
                        "topic_id": post.get('id'),
                        "title": post.get('title', 'No title'),
                        "created_at": post.get('created_at', 'Unknown'),
                        "url": post.get('url', 'No URL'),
                        "category_id": post.get('category_id'),
                        "tags": ', '.join(post.get('tags', [])) if post.get('tags') else '',
                        "image_count": len(post.get('images', []))
                    }
                    
                    kb.add_document(
                        text=post_text,
                        doc_id=f"discourse_topic_{post.get('id')}",
                        metadata=metadata
                    )
                    
                    posts_added += 1
                    
                    if posts_added % 10 == 0:
                        print(f"  Added {posts_added} posts...")
                        
                except Exception as e:
                    print(f"❌ Error processing post {post.get('id', 'unknown')}: {e}")
                    continue
            
            print(f"✅ Successfully added {posts_added} posts to knowledge base")
            
        except Exception as e:
            print(f"❌ Error loading discourse data: {e}")
    
    else:
        print(f"⚠️ {data_file_path} not found. Run the scraper first to generate discourse data.")
        
        # Check if file exists in current directory (fallback)
        fallback_path = "tds_discourse_posts.json"
        if os.path.exists(fallback_path):
            print(f"📚 Found {fallback_path} in current directory, loading from there...")
            
            try:
                with open(fallback_path, 'r', encoding='utf-8') as f:
                    discourse_data = json.load(f)
                
                print(f"Found {len(discourse_data)} posts in the fallback file")
                
                # Same processing logic as above
                posts_added = 0
                for post in discourse_data:
                    try:
                        post_text = f"""
Title: {post.get('title', 'No title')}
Created: {post.get('created_at', 'Unknown date')}
URL: {post.get('url', 'No URL')}
Category ID: {post.get('category_id', 'Unknown')}
Tags: {', '.join(post.get('tags', []))}

Full Content:
{post.get('full_text', 'No content')}

Individual Posts:
"""
                        
                        for i, individual_post in enumerate(post.get('posts', [])):
                            post_text += f"""
Post {i+1} by {individual_post.get('username', 'Unknown user')} at {individual_post.get('created_at', 'Unknown time')}:
{individual_post.get('text', 'No text')}
---
"""
                        
                        if post.get('images'):
                            post_text += "\nImages in this post:\n"
                            for img in post['images']:
                                post_text += f"- {img.get('alt', 'No description')}: {img.get('url', 'No URL')}\n"
                        
                        metadata = {
                            "source": "discourse",
                            "topic_id": post.get('id'),
                            "title": post.get('title', 'No title'),
                            "created_at": post.get('created_at', 'Unknown'),
                            "url": post.get('url', 'No URL'),
                            "category_id": post.get('category_id'),
                            "tags": ', '.join(post.get('tags', [])) if post.get('tags') else '',
                            "image_count": len(post.get('images', []))
                        }
                        
                        kb.add_document(
                            text=post_text,
                            doc_id=f"discourse_topic_{post.get('id')}",
                            metadata=metadata
                        )
                        
                        posts_added += 1
                        
                        if posts_added % 10 == 0:
                            print(f"  Added {posts_added} posts...")
                            
                    except Exception as e:
                        print(f"❌ Error processing post {post.get('id', 'unknown')}: {e}")
                        continue
                
                print(f"✅ Successfully added {posts_added} posts to knowledge base")
                
            except Exception as e:
                print(f"❌ Error loading fallback discourse data: {e}")
        else:
            print("📝 No discourse data found. Knowledge base will only contain course overview.")
    
    print("🎉 Setup complete!")
    
    # Test the knowledge base
    print("\n🧪 Testing knowledge base...")
    test_queries = [
        "What are the evaluation criteria for TDS?",
        "What models should I use for the project?",
        "Who are the teaching assistants?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        try:
            results = kb.search(query, n_results=3)
            print(f"   Found {len(results['documents'])} results")
            
            # Show first result snippet
            if results['documents'] and len(results['documents']) > 0:
                first_result = results['documents'][0]
                snippet = first_result[:200] + "..." if len(first_result) > 200 else first_result
                print(f"   First result: {snippet}")
        except Exception as e:
            print(f"   ❌ Error searching: {e}")
    
    print(f"\n📊 Knowledge base statistics:")
    try:
        # Try to get collection count if the method exists
        collection_info = kb.collection.count()
        print(f"   Total documents in knowledge base: {collection_info}")
    except:
        print("   Knowledge base ready for queries")

if __name__ == "__main__":
    main()
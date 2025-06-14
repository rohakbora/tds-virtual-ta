import requests
import json
from datetime import datetime
import time
from bs4 import BeautifulSoup
import os
import base64
from urllib.parse import urljoin, urlparse
import hashlib

class DiscourseScraperTDS:
    def __init__(self, base_url: str, cookies: dict):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        for name, value in cookies.items():
            self.session.cookies.set(name, value, domain=base_url.split('//')[1])
        
        # Create directories for storing images
        os.makedirs("images", exist_ok=True)
        os.makedirs("data", exist_ok=True)

    def verify_authentication(self):
        """Verify authentication with the Discourse API"""
        try:
            response = self.session.get(f"{self.base_url}/session/current.json")
            if response.status_code == 200:
                username = response.json().get("current_user", {}).get("username", "unknown")
                print(f"✅ Authenticated as: {username}")
                return True
            else:
                print(f"❌ Authentication failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False

    def download_image(self, img_url: str, topic_id: int) -> str:
        """Download and save image, return local path"""
        try:
            # Handle relative URLs
            if img_url.startswith('/'):
                img_url = urljoin(self.base_url, img_url)
            
            # Create filename from URL hash to avoid duplicates
            url_hash = hashlib.md5(img_url.encode()).hexdigest()[:8]
            parsed_url = urlparse(img_url)
            file_ext = os.path.splitext(parsed_url.path)[1] or '.jpg'
            filename = f"topic_{topic_id}_{url_hash}{file_ext}"
            filepath = os.path.join("images", filename)
            
            # Skip if already downloaded
            if os.path.exists(filepath):
                return filepath
            
            # Download image
            response = self.session.get(img_url, timeout=10)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"📸 Downloaded image: {filename}")
                return filepath
            else:
                print(f"❌ Failed to download image: {img_url}")
                return None
                
        except Exception as e:
            print(f"❌ Error downloading image {img_url}: {e}")
            return None

    def process_post_content(self, post_html: str, topic_id: int) -> dict:
        """Process post content, extract and download images"""
        soup = BeautifulSoup(post_html, "html.parser")
        images = []
        
        # Process images
        for img in soup.find_all("img"):
            img_url = img.get("src")
            alt_text = img.get("alt", "")
            
            if img_url:
                # Download image
                local_path = self.download_image(img_url, topic_id)
                
                image_info = {
                    "url": img_url,
                    "alt": alt_text,
                    "local_path": local_path
                }
                images.append(image_info)
                
                # Replace image with placeholder text
                placeholder = f"[Image: {alt_text or 'screenshot'}]"
                img.replace_with(placeholder)
        
        # Extract clean text
        clean_text = soup.get_text(separator="\n", strip=True)
        
        return {
            "text": clean_text,
            "images": images,
            "raw_html": post_html
        }

    def scrape_category_by_date(self, category_id: int, start_date: str, end_date: str, 
                               output_file="tds_discourse_posts.json"):
        """Scrape category posts with enhanced image handling"""
        all_data = []
        page = 0
        date_format = "%Y-%m-%d"
        start_dt = datetime.strptime(start_date, date_format)
        end_dt = datetime.strptime(end_date, date_format)
        
        print(f"🎯 Searching for posts between {start_date} and {end_date}")
        
        consecutive_empty_pages = 0
        max_consecutive_empty = 3
        
        while consecutive_empty_pages < max_consecutive_empty:
            try:
                url = f"{self.base_url}/c/{category_id}.json?page={page}"
                print(f"📄 Fetching page {page}...")
                response = self.session.get(url, timeout=15)
                
                if response.status_code != 200:
                    print(f"❌ Error fetching page {page}: {response.status_code}")
                    break

                data = response.json()
                topics = data.get("topic_list", {}).get("topics", [])
                
                if not topics:
                    print("✅ No more topics found. Done.")
                    break

                valid_posts_on_page = 0
                posts_too_old = 0
                
                for topic in topics:
                    topic_id = topic["id"]
                    slug = topic["slug"]
                    
                    # FIXED: Check date BEFORE fetching full topic details
                    # Parse the created_at from the topic list first
                    topic_created_at = datetime.strptime(topic["created_at"][:10], "%Y-%m-%d")
                    
                    if start_dt <= topic_created_at <= end_dt:
                        print(f"✅ Found valid post | {topic_created_at.date()} | {topic['title']}")
                        
                        # Only fetch full details and process images for posts in date range
                        full_topic = self.fetch_topic_details(topic_id, slug)
                        if full_topic:
                            all_data.append(full_topic)
                            valid_posts_on_page += 1
                    elif topic_created_at < start_dt:
                        print(f"⏪ Too old | {topic_created_at.date()} | {topic['title']}")
                        posts_too_old += 1
                    else:
                        print(f"⏩ Too new | {topic_created_at.date()} | {topic['title']}")

                # Update counters
                if valid_posts_on_page > 0:
                    consecutive_empty_pages = 0
                else:
                    consecutive_empty_pages += 1
                    
                print(f"📊 Page {page} summary: {valid_posts_on_page} valid posts, {posts_too_old} too old")
                
                if posts_too_old == len(topics) and posts_too_old > 0:
                    print(f"⚠️ All posts on page {page} are older than start date")
                    consecutive_empty_pages += 1

                page += 1
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error on page {page}: {e}")
                consecutive_empty_pages += 1
                page += 1
                time.sleep(2)

        print(f"🏁 Scraping completed. Found {len(all_data)} posts in total.")
        self.save_to_file(all_data, output_file)
        return all_data

    def fetch_topic_details(self, topic_id: int, slug: str) -> dict:
        """Fetch full topic including comments/replies with image processing"""
        try:
            url = f"{self.base_url}/t/{slug}/{topic_id}.json"
            response = self.session.get(url, timeout=15)
            
            if response.status_code != 200:
                print(f"❌ Failed to fetch topic {topic_id}: {response.status_code}")
                return None

            data = response.json()
            topic_url = f"{self.base_url}/t/{slug}/{topic_id}"

            processed_posts = []
            all_images = []
            
            for post in data.get("post_stream", {}).get("posts", []):
                # Process post content and extract images
                processed_content = self.process_post_content(post["cooked"], topic_id)
                
                processed_post = {
                    "id": post["id"],
                    "username": post["username"],
                    "created_at": post["created_at"],
                    "text": processed_content["text"],
                    "images": processed_content["images"],
                    "raw_html": processed_content["raw_html"]
                }
                
                processed_posts.append(processed_post)
                all_images.extend(processed_content["images"])

            # Combine all post texts
            full_text = "\n\n".join([p["text"] for p in processed_posts])

            return {
                "id": data["id"],
                "title": data["title"],
                "created_at": data["created_at"],
                "slug": slug,
                "url": topic_url,
                "posts": processed_posts,
                "full_text": full_text,
                "images": all_images,
                "category_id": data.get("category_id"),
                "tags": data.get("tags", [])
            }
            
        except Exception as e:
            print(f"❌ Error fetching topic {topic_id}: {e}")
            return None

    def save_to_file(self, data: list, filename: str):
        """Save scraped data to JSON file"""
        filepath = os.path.join("data", filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(data)} topics to {filepath}")

    def get_category_info(self, category_id: int):
        """Get category information"""
        try:
            response = self.session.get(f"{self.base_url}/c/{category_id}/show.json")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Error getting category info: {e}")
            return None

# Usage example
if __name__ == "__main__":
    # Replace with your actual cookies from browser
    browser_cookies = {
        '_t': 'MieY5BtDbSy3PzDRvD4rch%2FHQF%2BQUeqrgK%2FMfqfdvCjdsbfDiLGqks7pTm4nmqR3CzfwXlhqpoTF5yIWbEg3j6%2BFHcqtEVuN%2BXC3qUQl9UviSsZRBGAJDBsKU0RYlRbqlXkwS1L65APsMgd9Gzvh6vpb8lTteZBZLQ9RRzEgKjKRrep0gqylhCIWIYNa%2BjM82S2AZJUd4NlZvt5OJhXrt3H3IJvC54SwkgazYVo1qZnmitxqdPu58D2bQRI48o8x5c2bC145fldGog8YB5QohFoypKt1F65CiQ4gIDnTobuxg9b2EcrcEkzRk9KYjYK3--9xhS6t9BAWhSu8tR--89qj898q%2BSuUAbOjQg3EdA%3D%3D',
        '_forum_session': '1kG%2B9EhhuiCxkdygldPzRHHuxHgWC3tdyY%2Bx45kAQjbb%2B68X6dJRokU6eZyZLSeASWGkdZsyS6HSNajO6SGWR5ov%2FnNEeVRem9Ry0imANQBpqmZIGUjs54%2BdbKT1kYRQ6CuRA396Paw09tG10FHsh6in%2FXOI2c7j98nqmpGHgRieTAjPdY2Yb40wyfZjdcfbhvJk2bDOVyNOpeuD30kbO7TsZJeolCjnBD0KX3BtfRynpMTfHcsyrnNcZ9Ur7f%2F6rQGC9mwa8RPZs85q5H%2FSgesLb32MHg%3D%3D--dHk793ViQ1BX0n1x--6VXHgmuUqIit5BcswuBSCg%3D%3D'
    }

    scraper = DiscourseScraperTDS("https://discourse.onlinedegree.iitm.ac.in", browser_cookies)

    if scraper.verify_authentication():
        # Get category info first
        category_info = scraper.get_category_info(34)
        if category_info:
            print(f"Scraping category: {category_info.get('category', {}).get('name', 'Unknown')}")
        
        # Scrape posts
        posts = scraper.scrape_category_by_date(
            category_id=34,
            start_date="2025-01-01",
            end_date="2025-04-15"
        )
        
        print(f"✅ Scraping complete! Found {len(posts)} posts with images downloaded.")
    else:
        print("❌ Please update your cookies and try again.")
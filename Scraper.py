import requests
import json
import asyncio
from datetime import datetime
import time
from bs4 import BeautifulSoup
import os
from playwright.async_api import async_playwright

# ================== CONFIGURATION SECTION ==================
# Update these values as needed

# Discourse Configuration
DISCOURSE_BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
#Enter Cookies for authentication
DISCOURSE_COOKIES = {
    '_t': '',
    '_forum_session': ''
}
CATEGORY_ID = 34
START_DATE = "2025-01-01"
END_DATE = "2025-04-15"
DISCOURSE_OUTPUT_FILE = "Scraped_data.json"

# Docsify Configuration
DOCSIFY_BASE_URL = "https://tds.s-anand.net/"
DOCSIFY_OUTPUT_FILE = "scraped_website.jsonl"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50

# ================== DISCOURSE SCRAPER CLASS ==================

class DiscourseScraperTDS:
    def __init__(self, base_url: str, cookies: dict):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        for name, value in cookies.items():
            self.session.cookies.set(name, value, domain=base_url.split('//')[1])

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

    def process_post_content(self, post_html: str) -> str:
        """Process post content and extract clean text"""
        soup = BeautifulSoup(post_html, "html.parser")
        
        # Replace images with placeholder text
        for img in soup.find_all("img"):
            alt_text = img.get("alt", "")
            placeholder = f"[Image: {alt_text or 'screenshot'}]"
            img.replace_with(placeholder)
        
        # Extract clean text
        clean_text = soup.get_text(separator="\n", strip=True)
        return clean_text

    def scrape_category_by_date(self, category_id: int, start_date: str, end_date: str, 
                               output_file="Scraped_data.json"):
        """Scrape category posts for text content only"""
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
                    
                    # Check date BEFORE fetching full topic details
                    topic_created_at = datetime.strptime(topic["created_at"][:10], "%Y-%m-%d")
                    
                    if start_dt <= topic_created_at <= end_dt:
                        print(f"✅ Found valid post | {topic_created_at.date()} | {topic['title']}")
                        
                        # Only fetch full details for posts in date range
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
        """Fetch full topic including comments/replies"""
        try:
            url = f"{self.base_url}/t/{slug}/{topic_id}.json"
            response = self.session.get(url, timeout=15)
            
            if response.status_code != 200:
                print(f"❌ Failed to fetch topic {topic_id}: {response.status_code}")
                return None

            data = response.json()
            topic_url = f"{self.base_url}/t/{slug}/{topic_id}"

            processed_posts = []
            
            for post in data.get("post_stream", {}).get("posts", []):
                # Process post content to extract clean text
                clean_text = self.process_post_content(post["cooked"])
                
                processed_post = {
                    "id": post["id"],
                    "username": post["username"],
                    "created_at": post["created_at"],
                    "text": clean_text
                }
                
                processed_posts.append(processed_post)

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
                "category_id": data.get("category_id"),
                "tags": data.get("tags", [])
            }
            
        except Exception as e:
            print(f"❌ Error fetching topic {topic_id}: {e}")
            return None

    def save_to_file(self, data: list, filename: str):
        """Save scraped data to JSON file"""
        filepath = os.path.join(filename)
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

# ================== DOCSIFY SCRAPER FUNCTION ==================

async def scrape_docsify_site(base_url=DOCSIFY_BASE_URL, out_file=DOCSIFY_OUTPUT_FILE, 
                            chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Scrape Docsify site with configurable parameters"""
    
    async def wait_until_content_stable(page, selector, timeout=8000):
        last_text = None
        stable_time = 0
        while stable_time < 1000:
            try:
                text = await page.inner_text(selector)
            except:
                text = ""
            if text == last_text:
                stable_time += 200
            else:
                stable_time = 0
                last_text = text
            await page.wait_for_timeout(200)

    def chunk_content(text, size=chunk_size, overlap=overlap):
        chunks = []
        i = 0
        while i < len(text):
            end = i + size
            chunk = text[i:end]
            chunks.append(chunk)
            i = end - overlap
        return chunks

    data = []
    seen = set()

    print(f"🌍 Starting Docsify scrape of {base_url}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto(base_url)
        await page.wait_for_timeout(3000)
        await page.wait_for_selector("aside.sidebar a")

        links = await page.query_selector_all("aside.sidebar a")
        print(f"Found {len(links)} sidebar links")

        for link in links:
            href = await link.get_attribute("href")
            if not href or not href.startswith("#/"):
                continue
            full_url = base_url + href
            if full_url in seen:
                continue
            seen.add(full_url)

            await page.evaluate(f"window.location.hash = '{href}'")
            await wait_until_content_stable(page, "article.markdown-section")

            try:
                html = await page.inner_html("article.markdown-section")
                soup = BeautifulSoup(html, "html.parser")
                for details in soup.find_all("details"):
                    details["open"] = "true"
                text = soup.get_text(separator="\n").strip()

                for chunk in chunk_content(text):
                    data.append({"content": chunk, "url": full_url})
                print(f"✅ Scraped: {full_url} ({len(text)} chars, {len(chunk_content(text))} chunks)")
            except Exception as e:
                print(f"❌ Error at {full_url}: {e}")

        await browser.close()

    with open(out_file, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    print(f"\n✅ Saved {len(data)} entries to {out_file}")
    return data

# ================== MAIN EXECUTION ==================

if __name__ == "__main__":
    print("=" * 60)
    print("  DISCOURSE & DOCSIFY SCRAPER")
    print("=" * 60)
    print(f"📝 Configuration:")
    print(f"   Discourse URL: {DISCOURSE_BASE_URL}")
    print(f"   Category ID: {CATEGORY_ID}")
    print(f"   Date Range: {START_DATE} to {END_DATE}")
    print(f"   Docsify URL: {DOCSIFY_BASE_URL}")
    print(f"   Chunk Size: {CHUNK_SIZE} (overlap: {CHUNK_OVERLAP})")
    print("=" * 60)
    
    
    scraper = DiscourseScraperTDS(DISCOURSE_BASE_URL, DISCOURSE_COOKIES)

    if scraper.verify_authentication():
        # Get category info first
        category_info = scraper.get_category_info(CATEGORY_ID)
        if category_info:
            print(f"Scraping category: {category_info.get('category', {}).get('name', 'Unknown')}")
        
        # Scrape posts
        posts = scraper.scrape_category_by_date(
            CATEGORY_ID,
            start_date=START_DATE,
            end_date=END_DATE,
            output_file=DISCOURSE_OUTPUT_FILE
        )
        
        print(f"✅ Discourse scraping complete! Found {len(posts)} posts.")
    else:
        print("❌ Please update your cookies and try again.")

    print("\n🚀 Starting Docsify scraping...")
    try:
        docsify_data = asyncio.run(scrape_docsify_site())
        print(f"✅ Docsify scraping complete! Found {len(docsify_data)} chunks.")
    except Exception as e:
        print(f"❌ Docsify scraping failed: {e}")

    
    print("\n🎉 All scraping operations completed!")

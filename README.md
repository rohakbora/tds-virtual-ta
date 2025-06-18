# 🧠 TDS Virtual Teaching Assistant (IITM)

A fully functional AI assistant for the **Tools in Data Science** course (IITM BS Program).  
Includes:

- 📥 Discourse and Website scraping  
- 🔍 ChromaDB-powered semantic + hybrid search  
- ⚡ FastAPI backend for real-time QA  

---

## ⚡ Quick Start (No Scraping Needed)

This repository already includes the processed database files in `search_db`, as well as scraped data (`Scraped_data.json` and `scraped_website.jsonl`).  
You can **directly run the app** and start asking questions — no setup or scraping required.

#### AI Pipe API Key

Create a `.env` file in the root folder:

```bash
OPENAI_API_KEY='your_AI_Pipe_API_key'
```

### Run the FastAPI app:

```bash
pip install -r requirements.txt
uvicorn main:app
```

Visit: [http://localhost:8000/docs](http://localhost:8000/docs) to test the API.

Hosted version also available here:  
[https://rohakbora-tds-virtual-ta.hf.space/api/](https://rohakbora-tds-virtual-ta.hf.space/api/)

---

## 🛠 Requirements

```bash
python>=3.8
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📁 Directory Structure

| File / Dir              | Description                                       |
|-------------------------|---------------------------------------------------|
| `scraper.py`            | Scrapes Discourse + TDS website                   |
| `main.py`               | FastAPI app (auto-loads DB or triggers setup)    |
| `VectorDB.py`           | ChromaDB + SentenceTransformer-based search      |
| `setup.py`              | (Optional) Standalone index/test runner          |
| `Scraped_data.json`     | Forum post data (included)                       |
| `scraped_website.jsonl` | TDS website data (included)                      |
| `search_db/`            | Contains the ChromaDB database                   |

---

## 🌐 API Endpoints

Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

### `/api/` [POST] — Ask a question (optionally with image)

```json
{
  "question": "What tools are used in Project 1?",
  "image": "<base64 string>"  // optional
}
```

---

## 🔍 Search Features

- Semantic Search using `thenlper/gte-small`  
- Keyword Search  
- Hybrid Search  
- Auto-categorization (assignment, exam, course, etc.)  

---

## 💾 Data Sources

- 📘 `scraped_website.jsonl`: from [https://tds.s-anand.net/](https://tds.s-anand.net/)  
- 🗣️ `Scraped_data.json`: from [https://discourse.onlinedegree.iitm.ac.in/](https://discourse.onlinedegree.iitm.ac.in/)  

---

## 🔐 Authentication (for Discourse scraping)

If you wish to re-scrape the forum yourself, update `scraper.py` with your Discourse cookies:

```python
DISCOURSE_COOKIES = {
    "_t": "<your_token_here>",
    "_forum_session": "<your_session_cookie>"
}
```

---

## 🔄 Full Setup (Optional)

Only needed if you want to regenerate the database.

### Step 1: Scrape the Data

Install Playwright and run:

```bash
python -m playwright install
python scraper.py
```

Generates:

- `Scraped_data.json`  
- `scraped_website.jsonl`

### Step 2: Start the API

```bash
uvicorn main:app
```

If ChromaDB index doesn't exist, it will auto-create it from the above files.

---

## ❓ FAQ

**Do I need to run `setup.py`?**  
No — `main.py` handles everything. `setup.py` is optional for benchmarking or manual re-indexing.

**Can I test the search engine alone?**  
Yes — run:

```bash
python setup.py
```

---

## 🧠 Built For

The Tools in Data Science course in IITM's BSc in Data Science.  
This repo enables students to query course material, projects, exams, and forum discussions via natural language.

---

Let me know if you'd like:

- Docker support for deployment  
- A `.bat` or `.sh` launcher  
- Hugging Face Spaces compatibility  

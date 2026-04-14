# 🐸 Meme Finder

A RAG-powered meme search engine. Search your meme collection using plain English.

**Live Demo → [meme-finder on Streamlit](https://memeragtest-f5qg5isskxf2ec6vt3kva4.streamlit.app/)**

---

## How it works

```
Image → EasyOCR (reads text) → Groq LLM (understands meme) → FAISS (stores vectors) → Search
```

## Features
- Natural language search — *"dark humor"*, *"dad jokes"*, *"relatable"*
- OCR extracts meme text locally (no API cost)
- Duplicate detection via image hashing
- Metadata: title, category, emotion, funniness score
- Click any meme to view fullscreen

## Stack

| Tool | Purpose |
|------|---------|
| EasyOCR | Read text from images |
| Groq LLM | Analyze meme meaning |
| FAISS | Vector similarity search |
| Sentence Transformers | Text embeddings |
| Streamlit | Web UI |

## Run locally

```bash
git clone https://github.com/Muhammad-Shiraz/meme_rag_test
cd meme_rag_test
pip install -r requirements.txt
```

Add your Groq key to `.env`:
```
GROQ_API_KEY=your_key_here
```

```bash
streamlit run app.py
```

Get a free Groq key → [console.groq.com](https://console.groq.com)

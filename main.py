#!/usr/bin/env python3

import json
import re
import numpy as np
import faiss
import imagehash
import streamlit as st

from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer
from groq import Groq
import base64, io
import time
import pickle
# =========================
# CONFIG
# =========================
import os

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

groq_client = Groq(api_key=GROQ_API_KEY)
# st.sidebar.write("🔑 Key loaded:", GROQ_API_KEY[:8] + "...")
# MEME_FOLDER = r"C:\Users\Muhammad Shiraz\OneDrive\Desktop\Hackathon\memes"
MEME_FOLDER = "memes"

embedder = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# LOAD IMAGES
# =========================

def load_images(folder):
    folder = Path(folder)
    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
    images = []
    for e in exts:
        images += list(folder.rglob(e))
    return list(set(images))


# =========================
# DUPLICATE CHECK
# =========================

def is_duplicate(image_path, seen):
    try:
        img = Image.open(image_path)
        h = imagehash.average_hash(img)
        if h in seen:
            return True
        seen.add(h)
        return False
    except:
        return False


# =========================
# GEMINI ANALYSIS (Groq backend)
# =========================

PROMPT = """
You are a meme analyzer.

Return ONLY valid JSON:

{
  "text_in_image": "",
  "visual_description": "",
  "category": "",
  "emotion": "",
  "keywords": [],
  "summary": "",
  "title": "",
  "funniness": 5
}

funniness must be an integer from 1 (not funny) to 10 (extremely funny).
title should be a short, descriptive name for this meme.
"""


def safe_json_parse(text):
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except:
        return {
            "text_in_image": "",
            "visual_description": text[:200],
            "category": "general",
            "emotion": "unknown",
            "keywords": [],
            "summary": text[:200],
            "title": "",
            "funniness": 5
        }

def analyze_with_gemini(image_path):
    time.sleep(2)  # 2 seconds between each call
    
    for attempt in range(3):  # retry up to 3 times
        try:
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            b64 = base64.b64encode(buffer.getvalue()).decode()

            response = groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }],
                max_tokens=500
            )
            return safe_json_parse(response.choices[0].message.content)

        except Exception as e:
            error_msg = str(e)
            print(f"Attempt {attempt+1} failed: {error_msg}")
            
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                wait = (attempt + 1) * 10  # wait 10s, 20s, 30s
                print(f"Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                time.sleep(3)

    return {
        "text_in_image": "", "visual_description": "",
        "category": "general", "emotion": "unknown",
        "keywords": [], "summary": "failed",
        "title": "", "funniness": 5
    }


# =========================
# BUILD EMBEDDING TEXT
# =========================

def build_text(data):
    return " ".join([
        data.get("text_in_image", ""),
        data.get("visual_description", ""),
        data.get("category", ""),
        data.get("emotion", ""),
        " ".join(data.get("keywords", [])),
        data.get("summary", ""),
        data.get("title", ""),
        str(data.get("funniness", ""))
    ])


# =========================
# BUILD INDEX  (cached so it only runs once per session)
# =========================

@st.cache_resource(show_spinner=False)
def build_index(folder):
    import pickle

    # Load from disk if exists
    if Path("meme_index.faiss").exists() and Path("meme_meta.pkl").exists():
        st.info("Loading saved index...")
        index = faiss.read_index("meme_index.faiss")
        with open("meme_meta.pkl", "rb") as f:
            metadata = pickle.load(f)
        return index, metadata

    images = load_images(folder)
    vectors = []
    metadata = []
    seen = set()

    progress = st.progress(0, text="Building meme index…")
    total = len(images)

    for i, img in enumerate(images):
        progress.progress((i + 1) / max(total, 1), text=f"Processing {img.name} ({i+1}/{total})")

        if is_duplicate(img, seen):
            continue

        data = analyze_with_gemini(img)
        text = build_text(data)
        vec = embedder.encode(text).astype("float32")
        vectors.append(vec)
        metadata.append({"path": str(img), "data": data})

    progress.empty()

    if not vectors:
        return None, None

    vectors = np.array(vectors).astype("float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # Save to disk
    faiss.write_index(index, "meme_index.faiss")
    with open("meme_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)

    return index, metadata


# =========================
# SEARCH
# =========================

def search(query, index, metadata, k=6):
    q = embedder.encode(query).astype("float32").reshape(1, -1)
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, k)
    results = []
    for i, idx in enumerate(idxs[0]):
        if idx == -1:
            continue
        item = dict(metadata[idx])
        item["score"] = float(scores[0][i])
        results.append(item)
    return results


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Meme Finder", page_icon="🐸", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bangers&family=DM+Sans:wght@400;600&display=swap');

:root {
    --bg: #0e0e0e;
    --card: #1a1a1a;
    --accent: #f7c948;
    --accent2: #ff4d6d;
    --text: #f0f0f0;
    --muted: #888;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
}

[data-testid="stHeader"] { background: transparent !important; }

h1 { font-family: 'Bangers', cursive !important; font-size: 3.5rem !important;
     letter-spacing: 3px; color: var(--accent) !important; margin-bottom: 0 !important; }

.subtitle { font-family: 'DM Sans', sans-serif; color: var(--muted); font-size: 1rem; margin-bottom: 2rem; }

.meme-card {
    background: var(--card);
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 16px;
    transition: border-color .2s;
}
.meme-card:hover { border-color: var(--accent); }

.meme-title { font-family: 'Bangers', cursive; font-size: 1.3rem;
              color: var(--accent); letter-spacing: 1px; margin: 8px 0 4px; }

.badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.75rem; font-family: 'DM Sans', sans-serif;
    font-weight: 600; margin-right: 6px; margin-bottom: 4px;
}
.badge-cat  { background: #2a2a2a; color: var(--accent); }
.badge-emo  { background: #2a2a2a; color: #a78bfa; }
.badge-fun  { background: #2a2a2a; color: var(--accent2); }

.score-bar-wrap { background: #2a2a2a; border-radius: 6px; height: 6px; margin: 6px 0; }
.score-bar { background: var(--accent); border-radius: 6px; height: 6px; }

.summary-text { font-family: 'DM Sans', sans-serif; font-size: 0.85rem;
                color: var(--muted); margin-top: 6px; }

.stTextInput > div > div > input {
    background: #1a1a1a !important; color: var(--text) !important;
    border: 2px solid #333 !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 1rem !important;
    padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus { border-color: var(--accent) !important; }

.stButton > button {
    background: var(--accent) !important; color: #000 !important;
    font-family: 'Bangers', cursive !important; font-size: 1.1rem !important;
    letter-spacing: 1px; border: none !important; border-radius: 10px !important;
    padding: 10px 28px !important; cursor: pointer;
}
.stButton > button:hover { background: var(--accent2) !important; color: #fff !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1>MEME FINDER</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Search your meme stash with plain English</p>', unsafe_allow_html=True)

# Build / load index
with st.spinner("Loading meme index…"):
    index, metadata = build_index(MEME_FOLDER)

if index is None:
    st.error("No memes found in the folder. Check your MEME_FOLDER path.")
    st.stop()

st.success(f"✅ {len(metadata)} memes indexed and ready!")
# DEBUG
# import os
# st.write("Current dir:", os.getcwd())
# st.write("Files in current dir:", os.listdir("."))
# st.write("Memes folder exists:", Path("memes").exists())
# if metadata:
#     st.write("Saved path example:", metadata[0]["path"])

# Search bar
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input("", placeholder='Try: "dark humor", "dad jokes", "relatable office memes"…', label_visibility="collapsed")
with col2:
    search_btn = st.button("SEARCH 🔍")

k = st.slider("Results to show", 3, 12, 6)

if (search_btn or query) and query.strip():
    results = search(query.strip(), index, metadata, k=k)

    if not results:
        st.warning("No results found. Try a different query.")
    else:
        st.markdown(f"### Top {len(results)} memes for **\"{query}\"**")

        cols = st.columns(3)
        for i, r in enumerate(results):
            with cols[i % 3]:
                st.markdown('<div class="meme-card">', unsafe_allow_html=True)

                # # Show image
                # try:
                #     st.image(r["path"], use_container_width=True)
                # except:
                #     st.warning("Image not found")
                try:
                    img_name = Path(r["path"]).name
                    
                    # Try multiple possible locations
                    possible_paths = [
                        Path("memes") / img_name,
                        Path("/mount/src/meme-finder/memes") / img_name,
                        Path(r["path"]),
                    ]
                    
                    found = False
                    for p in possible_paths:
                        if p.exists():
                            st.image(str(p), use_container_width=True)
                            found = True
                            break
                    
                    if not found:
                        st.warning(f"❌ {img_name}")
                except Exception as e:
                    st.warning("Image error")

                d = r["data"]
                title    = d.get("title", "Untitled") or "Untitled"
                category = d.get("category", "?") or "?"
                emotion  = d.get("emotion", "?") or "?"
                funniness= d.get("funniness", "?")
                summary  = d.get("summary", "") or ""
                score    = r.get("score", 0)

                st.markdown(f'<div class="meme-title">{title}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<span class="badge badge-cat">📁 {category}</span>'
                    f'<span class="badge badge-emo">😶 {emotion}</span>'
                    f'<span class="badge badge-fun">😂 {funniness}/10</span>',
                    unsafe_allow_html=True
                )

                pct = int(score * 100)
                st.markdown(
                    f'<div class="score-bar-wrap"><div class="score-bar" style="width:{pct}%"></div></div>',
                    unsafe_allow_html=True
                )

                if summary:
                    st.markdown(f'<div class="summary-text">{summary}</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)












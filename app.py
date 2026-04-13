# #!/usr/bin/env python3

# import os, json, re, pickle
# import numpy as np
# import faiss
# import streamlit as st

# from pathlib import Path
# from PIL import Image
# from sentence_transformers import SentenceTransformer
# from groq import Groq

# # OCR + CLIP
# import easyocr
# import torch
# from transformers import CLIPProcessor, CLIPModel

# # =========================
# # CONFIG
# # =========================

# MEME_FOLDER = Path("memes").resolve()
# CACHE_FILE = "cache.pkl"

# groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# # OCR (EasyOCR)
# reader = easyocr.Reader(['en'])

# # CLIP
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # =========================
# # OCR
# # =========================

# def extract_text(image_path):
#     try:
#         result = reader.readtext(str(image_path))
#         return " ".join([r[1] for r in result])
#     except:
#         return ""

# # =========================
# # CLIP (image meaning)
# # =========================

# def get_clip_description(image_path):
#     try:
#         image = Image.open(image_path).convert("RGB")

#         labels = [
#             "funny meme",
#             "sad meme",
#             "reaction meme",
#             "angry meme",
#             "school meme",
#             "dark humor meme",
#             "relationship meme",
#             "gaming meme"
#         ]

#         inputs = clip_processor(
#             text=labels,
#             images=image,
#             return_tensors="pt",
#             padding=True
#         )

#         outputs = clip_model(**inputs)
#         probs = outputs.logits_per_image.softmax(dim=1)

#         return labels[probs.argmax()]

#     except:
#         return ""

# # =========================
# # CACHE
# # =========================

# def load_cache():
#     if os.path.exists(CACHE_FILE):
#         with open(CACHE_FILE, "rb") as f:
#             return pickle.load(f)
#     return {}

# def save_cache(cache):
#     with open(CACHE_FILE, "wb") as f:
#         pickle.dump(cache, f)

# # =========================
# # GROQ ANALYSIS
# # =========================

# def safe_json(text):
#     text = re.sub(r"```json|```", "", text)
#     try:
#         return json.loads(text)
#     except:
#         return {
#             "category": "general",
#             "emotion": "unknown",
#             "keywords": [],
#             "summary": text[:200],
#             "title": "Untitled",
#             "funniness": 5
#         }

# def analyze_with_groq(context):
#     prompt = f"""
#     Analyze this meme:

#     {context}

#     Return JSON:
#     {{
#       "category": "",
#       "emotion": "",
#       "keywords": [],
#       "summary": "",
#       "title": "",
#       "funniness": 1-10
#     }}
#     """

#     try:
#         res = groq_client.chat.completions.create(
#             model="llama3-8b-8192",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=200
#         )
#         return safe_json(res.choices[0].message.content)
#     except:
#         return {}

# # =========================
# # BUILD CONTEXT (OCR + CLIP)
# # =========================

# def build_context(image_path):
#     ocr_text = extract_text(image_path)
#     clip_text = get_clip_description(image_path)
#     filename = Path(image_path).stem

#     return f"""
# OCR TEXT: {ocr_text}
# IMAGE MEANING: {clip_text}
# FILENAME: {filename}
# """.strip()

# # =========================
# # ANALYZE IMAGE
# # =========================

# def analyze_image(image_path, cache):
#     name = Path(image_path).name

#     if name in cache:
#         return cache[name]

#     context = build_context(image_path)
#     data = analyze_with_groq(context)

#     cache[name] = data
#     save_cache(cache)

#     return data

# # =========================
# # LOAD IMAGES
# # =========================

# def load_images(folder):
#     exts = ["*.jpg", "*.png", "*.jpeg", "*.webp"]
#     images = []
#     for e in exts:
#         images += list(Path(folder).rglob(e))
#     return images

# # =========================
# # BUILD TEXT FOR SEARCH
# # =========================

# def build_text(data):
#     return " ".join([
#         data.get("summary", ""),
#         data.get("category", ""),
#         data.get("emotion", ""),
#         " ".join(data.get("keywords", [])),
#         data.get("title", ""),
#         str(data.get("funniness", ""))
#     ])

# # =========================
# # BUILD INDEX
# # =========================

# @st.cache_resource
# def build_index(folder):
#     cache = load_cache()

#     images = load_images(folder)

#     vectors = []
#     metadata = []

#     for img in images:
#         data = analyze_image(img, cache)
#         text = build_text(data)

#         vec = embedder.encode(text).astype("float32")
#         vectors.append(vec)

#         metadata.append({
#             "filename": img.name,
#             "data": data
#         })

#     vectors = np.array(vectors).astype("float32")
#     faiss.normalize_L2(vectors)

#     index = faiss.IndexFlatIP(vectors.shape[1])
#     index.add(vectors)

#     return index, metadata

# # =========================
# # SEARCH
# # =========================

# def search(query, index, metadata, k=6):
#     q = embedder.encode(query).astype("float32").reshape(1, -1)
#     faiss.normalize_L2(q)

#     scores, idxs = index.search(q, k)

#     results = []
#     for i, idx in enumerate(idxs[0]):
#         results.append({
#             **metadata[idx],
#             "score": float(scores[0][i])
#         })
#     return results

# # =========================
# # STREAMLIT UI
# # =========================

# st.title("😂 Meme Finder (OCR + CLIP + Groq)")

# index, metadata = build_index(MEME_FOLDER)

# query = st.text_input("Search memes")

# if query:
#     results = search(query, index, metadata)

#     for r in results:
#         st.image(str(MEME_FOLDER / r["filename"]))
#         st.write("**Result:**", r["data"])
#         st.write("Score:", r["score"])
#         st.divider()

















#!/usr/bin/env python3

# ================================
# IMPORTS
# ================================
import os
import json
import re
import pickle
import time
import base64
import io
import numpy as np
import faiss
import imagehash
import streamlit as st

from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer
from groq import Groq

# OCR - reads text from images locally (no API needed, no installation needed)
import easyocr

# Load EasyOCR once (English language)
ocr_reader = easyocr.Reader(['en'], gpu=False)

# ================================
# CONFIG
# ================================

# Groq API key - reads from Streamlit secrets on cloud, env variable locally
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

groq_client = Groq(api_key=GROQ_API_KEY)

# Meme folder - works on both local and Streamlit Cloud
MEME_FOLDER = "memes"

# Files to save the index so we don't rebuild every time
INDEX_FILE = "meme_index.faiss"
META_FILE  = "meme_meta.pkl"

# Sentence embedder for search
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ================================
# STEP 1 — LOAD ALL IMAGES
# ================================

def load_images(folder):
    """Find all image files in the memes folder"""
    folder = Path(folder)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]

    images = []
    for ext in extensions:
        images += list(folder.rglob(ext))

    return list(set(images))


# ================================
# STEP 2 — SKIP DUPLICATE IMAGES
# ================================

def is_duplicate(image_path, seen_hashes):
    """Check if we already have this image (even if filename is different)"""
    try:
        img = Image.open(image_path)
        h = imagehash.average_hash(img)  # fingerprint of the image

        if h in seen_hashes:
            return True  # already seen this image

        seen_hashes.add(h)
        return False
    except:
        return False


# ================================
# STEP 3 — OCR: READ TEXT FROM IMAGE
# ================================

def extract_text_with_ocr(image_path):
    """Use EasyOCR to read any text inside the meme image — runs locally, no API cost"""
    try:
        results = ocr_reader.readtext(str(image_path))
        # Each result is [bbox, text, confidence] — we just want the text
        text = " ".join([r[1] for r in results]).strip()
        return text[:300]  # limit to 300 characters to save tokens
    except:
        return ""  # if OCR fails, return empty string


# ================================
# STEP 4 — GROQ LLM: ANALYZE MEME
# ================================

def analyze_with_groq(image_path, ocr_text):
    """
    Send image + OCR text to Groq LLM.
    OCR already got the text so Groq only needs to understand the visual.
    This saves ~60% tokens compared to asking Groq to do everything.
    """

    # Build prompt — include OCR text so Groq doesn't waste tokens re-reading it
    prompt = f"""You are a meme analyzer.

The text already extracted from this meme via OCR is:
"{ocr_text}"

Look at the image and return ONLY this JSON (no extra text, no markdown):
{{
  "text_in_image": "{ocr_text}",
  "visual_description": "",
  "category": "",
  "emotion": "",
  "keywords": [],
  "summary": "",
  "title": "",
  "funniness": 5
}}

- category: type of meme (dark humor, relatable, dad joke, political, etc.)
- emotion: main feeling (funny, sad, angry, wholesome, etc.)
- keywords: 3-5 relevant search words as a list
- summary: one sentence explaining the joke or meaning
- title: short catchy name for this meme
- funniness: integer 1 (boring) to 10 (hilarious)
"""

    # Try up to 3 times in case of temporary rate limiting
    for attempt in range(3):
        try:
            # Convert image to base64 so Groq can see it
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            b64_image = base64.b64encode(buffer.getvalue()).decode()

            # Call Groq API
            response = groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                    ]
                }],
                max_tokens=200  # reduced because OCR already got the text
            )

            result = safe_json_parse(response.choices[0].message.content)

            # Always keep OCR text even if Groq missed it
            if not result.get("text_in_image") and ocr_text:
                result["text_in_image"] = ocr_text

            return result

        except Exception as e:
            error_msg = str(e)
            print(f"Attempt {attempt + 1} failed: {error_msg}")

            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                wait_seconds = (attempt + 1) * 10  # 10s, 20s, 30s
                print(f"Rate limited — waiting {wait_seconds}s...")
                time.sleep(wait_seconds)
            else:
                time.sleep(3)

    # If all 3 attempts fail — still return OCR text so meme is not lost
    return {
        "text_in_image":      ocr_text,
        "visual_description": "",
        "category":           "general",
        "emotion":            "unknown",
        "keywords":           ocr_text.split()[:5] if ocr_text else [],
        "summary":            ocr_text[:100] if ocr_text else "failed",
        "title":              Path(image_path).stem.replace("-", " ").title(),
        "funniness":          5
    }


# ================================
# HELPER — SAFELY PARSE JSON
# ================================

def safe_json_parse(text):
    """Clean up Groq response and parse as JSON"""
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except:
        return {
            "text_in_image":      "",
            "visual_description": text[:200],
            "category":           "general",
            "emotion":            "unknown",
            "keywords":           [],
            "summary":            text[:200],
            "title":              "",
            "funniness":          5
        }


# ================================
# STEP 5 — BUILD SEARCH TEXT
# ================================

def build_search_text(data):
    """Combine all meme info into one string for the search embedder"""
    return " ".join([
        data.get("text_in_image",      ""),
        data.get("visual_description", ""),
        data.get("category",           ""),
        data.get("emotion",            ""),
        " ".join(data.get("keywords",  [])),
        data.get("summary",            ""),
        data.get("title",              ""),
        str(data.get("funniness",      ""))
    ])


# ================================
# STEP 6 — BUILD FAISS INDEX
# ================================

@st.cache_resource(show_spinner=False)
def build_index(folder):
    """
    Builds the searchable index.
    - If saved index exists on disk → loads it instantly
    - Otherwise → processes all memes one by one and saves the result
    """

    # Load saved index if it exists (fast path)
    if Path(INDEX_FILE).exists() and Path(META_FILE).exists():
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata

    # Build fresh index (slow path — only runs once)
    images      = load_images(folder)
    vectors     = []
    metadata    = []
    seen_hashes = set()

    progress_bar = st.progress(0, text="Building meme index…")
    total        = len(images)

    for i, img_path in enumerate(images):

        progress_bar.progress(
            (i + 1) / max(total, 1),
            text=f"Processing {img_path.name} ({i+1}/{total})"
        )

        # Skip duplicate images
        if is_duplicate(img_path, seen_hashes):
            print(f"Skipping duplicate: {img_path.name}")
            continue

        # STEP A: OCR reads text from image (free, local, fast)
        ocr_text = extract_text_with_ocr(img_path)

        # STEP B: Groq analyzes image (uses OCR text to save tokens)
        time.sleep(2)  # 2 second pause to avoid rate limiting
        data = analyze_with_groq(img_path, ocr_text)

        # STEP C: Create search vector
        search_text = build_search_text(data)
        vector      = embedder.encode(search_text).astype("float32")

        vectors.append(vector)
        metadata.append({
            "path":     str(img_path),
            "filename": img_path.name,
            "data":     data
        })

    progress_bar.empty()

    if not vectors:
        return None, None

    # Build FAISS vector index
    vectors = np.array(vectors).astype("float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # Save to disk so next run loads instantly
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)

    return index, metadata


# ================================
# STEP 7 — SEARCH FUNCTION
# ================================

def search(query, index, metadata, k=6):
    """Search memes using natural language"""
    query_vector = embedder.encode(query).astype("float32").reshape(1, -1)
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        item          = dict(metadata[idx])
        item["score"] = float(scores[0][i])
        results.append(item)

    return results


# ================================
# STREAMLIT UI
# ================================

st.set_page_config(page_title="Meme Finder 🐸", page_icon="🐸", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bangers&family=DM+Sans:wght@400;600&display=swap');
:root{--bg:#0e0e0e;--card:#1a1a1a;--accent:#f7c948;--accent2:#ff4d6d;--text:#f0f0f0;--muted:#888;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;}
[data-testid="stHeader"]{background:transparent!important;}
h1{font-family:'Bangers',cursive!important;font-size:3.5rem!important;letter-spacing:3px;color:var(--accent)!important;}
.meme-card{background:var(--card);border:1px solid #2a2a2a;border-radius:12px;padding:12px;margin-bottom:16px;}
.meme-card:hover{border-color:var(--accent);}
.meme-title{font-family:'Bangers',cursive;font-size:1.3rem;color:var(--accent);letter-spacing:1px;margin:8px 0 4px;}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:0.75rem;font-weight:600;margin-right:6px;}
.badge-cat{background:#2a2a2a;color:var(--accent);}
.badge-emo{background:#2a2a2a;color:#a78bfa;}
.badge-fun{background:#2a2a2a;color:var(--accent2);}
.score-bar-wrap{background:#2a2a2a;border-radius:6px;height:6px;margin:6px 0;}
.score-bar{background:var(--accent);border-radius:6px;height:6px;}
.summary-text{font-size:0.85rem;color:var(--muted);margin-top:6px;}
.stTextInput>div>div>input{background:#1a1a1a!important;color:var(--text)!important;border:2px solid #333!important;border-radius:10px!important;padding:12px 16px!important;}
.stButton>button{background:var(--accent)!important;color:#000!important;font-family:'Bangers',cursive!important;font-size:1.1rem!important;border:none!important;border-radius:10px!important;padding:10px 28px!important;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1>🐸 MEME FINDER</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#888">Search your meme stash with plain English</p>', unsafe_allow_html=True)

# Build or load index
with st.spinner("Loading meme index…"):
    index, metadata = build_index(MEME_FOLDER)

if index is None:
    st.error("No memes found! Check your memes folder.")
    st.stop()

st.success(f"✅ {len(metadata)} memes ready!")

# Search bar
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input("Search", placeholder='Try: "dark humor", "dad jokes", "relatable"…', label_visibility="collapsed")
with col2:
    search_btn = st.button("SEARCH 🔍")

k = st.slider("Results to show", 3, 12, 6)

# Show results
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

                # Show image — works on local and Streamlit Cloud
                try:
                    img_name = Path(r["path"]).name
                    img_path = Path("memes") / img_name
                    if img_path.exists():
                        st.image(str(img_path), use_container_width=True)
                    else:
                        st.warning(f"❌ {img_name}")
                except:
                    st.warning("Image error")

                # Show metadata
                d         = r["data"]
                title     = d.get("title",     "Untitled") or "Untitled"
                category  = d.get("category",  "?")        or "?"
                emotion   = d.get("emotion",   "?")        or "?"
                funniness = d.get("funniness", "?")
                summary   = d.get("summary",   "")         or ""
                score     = r.get("score", 0)

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
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
import easyocr

ocr_reader = easyocr.Reader(['en'], gpu=False)

# ================================
# CONFIG
# ================================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    # GROQ_API_KEY = ""

groq_client = Groq(api_key=GROQ_API_KEY)

MEME_FOLDER = "memes"
INDEX_FILE  = "meme_index.faiss"
META_FILE   = "meme_meta.pkl"

embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ================================
# LOAD IMAGES
# ================================
def load_images(folder):
    folder = Path(folder)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
    images = []
    for ext in extensions:
        images += list(folder.rglob(ext))
    return list(set(images))


# ================================
# DUPLICATE CHECK
# ================================
def is_duplicate(image_path, seen_hashes):
    try:
        img = Image.open(image_path)
        h = imagehash.average_hash(img)
        if h in seen_hashes:
            return True
        seen_hashes.add(h)
        return False
    except:
        return False


# ================================
# OCR
# ================================
def extract_text_with_ocr(image_path):
    try:
        results = ocr_reader.readtext(str(image_path))
        text = " ".join([r[1] for r in results]).strip()
        return text[:300]
    except:
        return ""


# ================================
# GROQ ANALYSIS (FIXED + RETRY)
# ================================
def analyze_with_groq(image_path, ocr_text):

    prompt = f"""
You are a meme analyzer.

Meme text:
"{ocr_text}"

Return ONLY JSON:
{{
  "text_in_image": "{ocr_text}",
  "visual_description": "short description",
  "category": "relatable/dark/dad joke/etc",
  "emotion": "funny/sad/angry/etc",
  "keywords": ["k1", "k2", "k3"],
  "summary": "one sentence meaning",
  "title": "short title",
  "funniness": 1-10
}}
"""

    for attempt in range(3):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # ✅ GOOD MODEL
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )

            return safe_json_parse(response.choices[0].message.content)

        except Exception as e:
            print(f"❌ GROQ ERROR (attempt {attempt+1}):", e)
            time.sleep(3)

    print("⚠️ Using fallback metadata")

    return {
        "text_in_image": ocr_text,
        "visual_description": "",
        "category": "general",
        "emotion": "unknown",
        "keywords": ocr_text.split()[:5] if ocr_text else [],
        "summary": ocr_text[:100],
        "title": Path(image_path).stem,
        "funniness": 5
    }

# ================================
# JSON PARSER
# ================================
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


# ================================
# BUILD SEARCH TEXT
# ================================
def build_search_text(data):
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


# ================================
# BUILD INDEX (GRADIO STYLE PATH)
# ================================
@st.cache_resource(show_spinner=False)
def build_index(folder):

    if Path(INDEX_FILE).exists() and Path(META_FILE).exists():
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata

    images = load_images(folder)
    vectors = []
    metadata = []
    seen_hashes = set()

    for img_path in images:

        if is_duplicate(img_path, seen_hashes):
            continue

        ocr_text = extract_text_with_ocr(img_path)
        data = analyze_with_groq(img_path, ocr_text)

        search_text = build_search_text(data)
        vector = embedder.encode(search_text).astype("float32")

        vectors.append(vector)

        # 🔥 GRADIO STYLE PATH
        metadata.append({
            "path": str(img_path.as_posix()),  # 🔥 IMPORTANT FIX,   # e.g. memes/funny.jpg
            "data": data
        })

    if not vectors:
        return None, None

    vectors = np.array(vectors).astype("float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)

    return index, metadata


# ================================
# SEARCH
# ================================
def search(query, index, metadata, k=6):
    query_vector = embedder.encode(query).astype("float32").reshape(1, -1)
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        item = dict(metadata[idx])
        item["score"] = float(scores[0][i])
        results.append(item)

    return results


# ================================
# STREAMLIT UI
# ================================
st.set_page_config(page_title="Meme Finder 🐸", layout="wide")

st.title("🐸 Meme Finder")

index, metadata = build_index(MEME_FOLDER)

query = st.text_input("Search memes")

if query:
    results = search(query, index, metadata)

    cols = st.columns(3)

    for i, r in enumerate(results):
        with cols[i % 3]:
            img_path = r.get("path", "").replace("\\", "/")
            st.write("DEBUG PATH:", img_path)

            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.warning(f"Missing: {img_path}")

            d = r["data"]

            st.markdown(f"**{d.get('title', 'Untitled')}**")
            st.write(f"📂 {d.get('category')} | 😂 {d.get('funniness')}/10")
            st.write(d.get("summary", ""))
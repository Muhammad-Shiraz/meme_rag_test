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
# GROQ ANALYSIS
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
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )
            return safe_json_parse(response.choices[0].message.content)

        except Exception as e:
            print(f"GROQ ERROR (attempt {attempt+1}):", e)
            time.sleep(3)

    return {
        "text_in_image":      ocr_text,
        "visual_description": "",
        "category":           "general",
        "emotion":            "unknown",
        "keywords":           ocr_text.split()[:5] if ocr_text else [],
        "summary":            ocr_text[:100],
        "title":              Path(image_path).stem,
        "funniness":          5
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
# BUILD SEARCH TEXT
# ================================
def build_search_text(data):
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
# BUILD INDEX
# ================================
@st.cache_resource(show_spinner=False)
def build_index(folder):

    if Path(INDEX_FILE).exists() and Path(META_FILE).exists():
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata

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

        if is_duplicate(img_path, seen_hashes):
            continue

        ocr_text    = extract_text_with_ocr(img_path)
        data        = analyze_with_groq(img_path, ocr_text)
        search_text = build_search_text(data)
        vector      = embedder.encode(search_text).astype("float32")

        vectors.append(vector)
        metadata.append({
            "path": str(img_path.as_posix()),  # always forward slashes e.g. memes/funny.jpg
            "data": data
        })

    progress_bar.empty()

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
        item          = dict(metadata[idx])
        item["score"] = float(scores[0][i])
        results.append(item)

    return results


# ================================
# IMAGE TO BASE64
# ================================
def image_to_base64(img_path):
    """Embed image directly in HTML — no file path needed on cloud"""
    try:
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=75)
        return base64.b64encode(buffer.getvalue()).decode()
    except:
        return ""


# ================================
# FIND IMAGE
# ================================
def find_image(saved_path):
    """
    Find image file on any platform.
    saved_path is like: memes/funny.jpg  (posix format)
    """
    img_name = Path(saved_path).name

    candidates = [
        saved_path,                        # direct saved path
        f"memes/{img_name}",               # relative
        str(Path("memes") / img_name),     # Path object
    ]

    # Auto-detect Streamlit Cloud /mount/src/<repo>/memes/
    mount_base = Path("/mount/src")
    if mount_base.exists():
        for repo_dir in mount_base.iterdir():
            candidates.append(str(repo_dir / "memes" / img_name))

    for c in candidates:
        if c and os.path.exists(c):
            return c

    return None


# ================================
# STREAMLIT UI
# ================================

st.set_page_config(page_title="Meme Finder 🐸", page_icon="🐸", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bangers&family=DM+Sans:wght@400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0e0e0e !important;
    color: #f0f0f0 !important;
}
[data-testid="stHeader"] { background: transparent !important; }

.meme-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin-top: 16px;
}

.meme-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    overflow: hidden;
    transition: border-color 0.2s;
}
.meme-card:hover { border-color: #f7c948; }

.meme-card img {
    width: 100%;
    height: 200px;
    object-fit: contain;
    display: block;
    background: #111;
    cursor: pointer;
}

.meme-info {
    padding: 8px 10px 10px;
}

.meme-title {
    font-family: 'Bangers', cursive;
    font-size: 15px;
    color: #f7c948;
    letter-spacing: 0.5px;
    margin-bottom: 5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-bottom: 5px;
}

.badge {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 20px;
    background: #2a2a2a;
}
.badge-cat { color: #f7c948; }
.badge-emo { color: #a78bfa; }
.badge-fun { color: #ff4d6d; }

.score-bar-wrap {
    background: #2a2a2a;
    border-radius: 4px;
    height: 3px;
    margin: 4px 0;
}
.score-bar {
    background: #f7c948;
    border-radius: 4px;
    height: 3px;
}

.summary {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    color: #888;
    margin-top: 4px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.stTextInput > div > div > input {
    background: #1a1a1a !important;
    color: #f0f0f0 !important;
    border: 2px solid #333 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #f7c948 !important;
}
.stButton > button {
    background: #f7c948 !important;
    color: #000 !important;
    font-family: 'Bangers', cursive !important;
    font-size: 1.1rem !important;
    letter-spacing: 1px;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
}
.stButton > button:hover {
    background: #ff4d6d !important;
    color: #fff !important;
}
            

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; padding: 24px 0 8px;">
  <span style="font-family:'Bangers',cursive; font-size:3rem; color:#f7c948; letter-spacing:3px;">
    🐸 MEME FINDER
  </span><br>
  <span style="font-family:'DM Sans',sans-serif; color:#888; font-size:0.95rem;">
    Search your meme stash with plain English
  </span>
</div>
""", unsafe_allow_html=True)

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

k = st.slider("Results to show", 4, 16, 8)

# Show results
if (search_btn or query) and query.strip():
    results = search(query.strip(), index, metadata, k=k)

    if not results:
        st.warning("No results found. Try a different query.")
    else:
        st.markdown(
            f"<p style='color:#888; font-family:DM Sans,sans-serif; font-size:0.85rem;'>"
            f"Top {len(results)} results for <b style='color:#f7c948'>\"{query}\"</b></p>",
            unsafe_allow_html=True
        )

        cards_html = '<div class="meme-grid">'

        for r in results:
            d         = r["data"]
            title     = (d.get("title",    "Untitled") or "Untitled")[:40]
            category  = d.get("category",  "?") or "?"
            emotion   = d.get("emotion",   "?") or "?"
            funniness = d.get("funniness", 5)
            summary   = d.get("summary",   "") or ""
            score     = int(r.get("score", 0) * 100)

            # Find image using saved path + fallbacks
            saved_path = r.get("path", "")
            img_found  = find_image(saved_path)

            if img_found:
                b64 = image_to_base64(img_found)
                cards_html += f"""
                <div class="meme-card">
                  <img src="data:image/jpeg;base64,{b64}" alt="{title}" onclick="window.open(this.src,'_blank')">
                  <div class="meme-info">
                    <div class="meme-title">{title}</div>
                    <div class="badge-row">
                      <span class="badge badge-cat">📁 {category}</span>
                      <span class="badge badge-emo">😶 {emotion}</span>
                      <span class="badge badge-fun">😂 {funniness}/10</span>
                    </div>
                    <div class="score-bar-wrap">
                      <div class="score-bar" style="width:{score}%"></div>
                    </div>
                    <div class="summary">{summary}</div>
                  </div>
                </div>"""
            else:
                img_name = Path(saved_path).name
                cards_html += f"""
                <div class="meme-card">
                  <div class="meme-info">
                    <div class="meme-title">{title}</div>
                    <div style="color:#666; font-size:11px; padding:8px 0;">❌ {img_name}</div>
                  </div>
                </div>"""

        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)
        
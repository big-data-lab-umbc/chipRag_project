import os
import re
import time

import torch
import numpy as np
import pandas as pd
import faiss

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Config
# ============================================================
MODEL_NAME  = "Qwen/Qwen2.5-3B-Instruct"
EMBED_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
RT_CSV_PATH = "/umbc/rs/pi_jianwu/users/tartelt1/CHIPRAG/allDoitTickets_CLEANED.csv"
CACHE_DIR   = "rag_cache"
LOG_FILE    = "Qwen_rt_v1.txt"

COL_SUBJECT = "SubjectNoHTML"
COL_BODY    = "TransactionContent"

TOP_K = 3
MAX_NEW_TOKENS = 768
CHUNK_CHARS = 1500

# ============================================================
# GPU check
# ============================================================
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    DEVICE = "cuda"
    DTYPE  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    print("No GPU found. Model will run on CPU (slow).")
    DEVICE = "cpu"
    DTYPE  = torch.float32

# ============================================================
# LLM
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    dtype=DTYPE,                 # torch_dtype is deprecated
)
model.eval()

#==============================================================
#Sytem promt
#==============================================================

SYSTEM = """You are a support assistant for the UMBC chip HPC cluster.
Use only the retrieved institutional evidence for site-specific facts.
Do not invent commands, paths, partitions, accounts, policies, or system details.
Give safe, step-by-step instructions when the evidence supports them.
If the evidence is insufficient, clearly state what information is missing.
Provide a detailed answer with at least 5 numbered steps.
Explain each step clearly in 2-3 sentences.
Always complete your response fully. Do not stop mid-sentence.
"""

def answer_llm(user_content: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Single generation entry point. SYSTEM is applied here ONCE, via ChatML."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": user_content},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,     # deterministic; must match the baseline script
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated = out[0][inputs["input_ids"].shape[-1]:]   # slice TOKENS, not chars
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ============================================================
# RT preprocessing
# ============================================================
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
USERPATH_RE = re.compile(r"/(?:users|home|nfs/users)/[^/\s]+")

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def anonymize_text(s: str) -> str:
    s = normalize_text(s)
    s = EMAIL_RE.sub("[EMAIL]", s)
    s = PHONE_RE.sub("[PHONE]", s)
    s = IP_RE.sub("[IP]", s)
    s = USERPATH_RE.sub(lambda m: f"/{m.group(0).split('/')[1]}/[USER]", s)
    return s


df = pd.read_csv(RT_CSV_PATH)
print("DataFrame columns:", df.columns.tolist())

for col in (COL_SUBJECT, COL_BODY):
    if col not in df.columns:
        raise KeyError(f"Expected column {col!r} not found in {RT_CSV_PATH}")

# fillna BEFORE astype(str), otherwise NaN becomes the literal string "nan"
df["subject_clean"] = df[COL_SUBJECT].fillna("").astype(str).map(anonymize_text)
df["body_clean"]    = df[COL_BODY].fillna("").astype(str).map(anonymize_text)
df["problem_text"]  = (df["subject_clean"] + "\n\n" + df["body_clean"]).str.slice(0, 4000)

# drop empties and exact duplicates (RT stores one row per transaction)
df = df[df["problem_text"].str.strip().str.len() > 20]
df = df.drop_duplicates(subset="problem_text").reset_index(drop=True)

#print("Rows after cleaning:", len(df))
#print("\nExample anonymized ticket:")
#print(df.iloc[0]["body_clean"][:400])          # .iloc, not .loc[0]

# ============================================================
# FAISS retrieval (cached)
# ============================================================
os.makedirs(CACHE_DIR, exist_ok=True)
INDEX_PATH = os.path.join(CACHE_DIR, "rt_tickets.faiss")
META_PATH  = os.path.join(CACHE_DIR, "rt_tickets_meta.parquet")

embedder = SentenceTransformer(EMBED_NAME, device=DEVICE)

if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    print("Loading cached FAISS index...")
    index = faiss.read_index(INDEX_PATH)
    df = pd.read_parquet(META_PATH)
    assert index.ntotal == len(df), "Cache is stale -- delete rag_cache/ and rerun."
else:
    print("Encoding tickets (first run only)...")
    embs = embedder.encode(
        df["problem_text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(embs.shape[1])   # cosine, since embs are normalized
    index.add(embs)

    faiss.write_index(index, INDEX_PATH)
    df.to_parquet(META_PATH)
    print(f"Cached index ({index.ntotal} vectors) to {CACHE_DIR}/")


def retrieve_rt(query: str, top_k: int = TOP_K):
    q_emb = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, idxs = index.search(q_emb, top_k)

    hits = []

    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue

        hits.append({
            "score": float(score),
            "subject": df.iloc[idx]["subject_clean"],
            "text": df.iloc[idx]["problem_text"][:CHUNK_CHARS],
        })

    return hits


def build_rag_prompt(query: str, hits: list[dict]) -> str:
    """Returns the USER message only -- SYSTEM is added by answer_llm via ChatML."""

    if not hits:
        return (
            f"User question:\n{query}\n\n"
            "No relevant past tickets were found in the knowledge base.\n"
            "Answer from general HPC knowledge if possible, but state clearly that "
            "no site-specific RT ticket evidence supported the answer.\n"
        )

    # similarity scores deliberately excluded -- the model parrots them.
    context = "\n\n---\n\n".join(
        f"[Ticket {i}]\nSubject: {h['subject']}\nText: {h['text']}"
        for i, h in enumerate(hits, 1)
    )

    return (
        f"User question:\n{query}\n\n"
        f"Evidence retrieved from the RT ticket knowledge base:\n{context}\n\n"
        "Use the retrieved evidence when relevant.\n"
        "If the evidence is insufficient, say what information is missing."
    )

def answer_rag(question: str):
    hits = retrieve_rt(question)
    user_content = build_rag_prompt(question, hits)
    return answer_llm(user_content), hits


def log_qa(question: str, answer: str, hits: list[dict], elapsed: float) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Question:\n{question}\n\n")
        f.write(f"\nAnswer:\n{answer}\n")


# ============================================================
# Interactive chatbot
# ============================================================
print("\nQwen 2.5 3B + Ticket RAG is ready.")
print("Type your chip/HPC question. Type 'exit' to quit.")

while True:
    try:
        question = input("\nAsk a CHIP/HPC question: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting Qwen 2.5 3B + Ticket RAG chatbot.")
        break

    if not question:
        print("Please type a question, or type 'exit' to quit.")
        continue
    if question.lower() in {"exit", "quit"}:
        print("Exiting Qwen 2.5 3B + Ticket RAG chatbot.")
        break

    print(f"\nQuestion: {question}")          # explicit echo for batch logs
    print("--- Qwen 2.5 3B + Ticket RAG answer ---")

    t0 = time.time()
    rag_answer, hits = answer_rag(question)
    elapsed = time.time() - t0

    print(rag_answer)
    print(f"\n[{elapsed:.1f}s | {len(hits)} tickets retrieved]")

    log_qa(question, rag_answer, hits, elapsed)

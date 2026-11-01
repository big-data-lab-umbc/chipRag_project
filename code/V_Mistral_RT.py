import os
import re
import torch
import pandas as pd
import numpy as np
import faiss

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Configuration

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

RT_CSV_PATH = (
#    "/choose user appropriate path/"
    "allDoitTickets_CLEANED.csv"
)

COL_SUBJECT = "SubjectNoHTML"
COL_BODY = "TransactionContent"

TOP_K = 3
CHUNK_CHARS = 1500
MAX_NEW_TOKENS = 768

LOG_PATH = "V1_rt_mistral.txt"

# GPU check

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("No GPU found. Model will run on CPU.")

# Load Mistral model

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
)

model.eval()

print("Loaded model:", MODEL_NAME)
print("Model config:", model.config._name_or_path)

# System prompt
SYSTEM = """You are a support assistant for the UMBC chip HPC cluster.
Use only the retrieved institutional evidence for site-specific facts.
Do not invent commands, paths, partitions, accounts, policies, or system details.
Give safe, step-by-step instructions when the evidence supports them.
If the evidence is insufficient, clearly state what information is missing.
Provide a detailed answer with at least 5 numbered steps.
Explain each step clearly in 2-3 sentences.
Always complete your response fully. Do not stop mid-sentence.
"""
# Generic generation function

def generate_from_prompt(
    user_content: str,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> str:

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_content},
   ] 

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = out[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()

# ============================================================
# RT preprocessing

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

PHONE_RE = re.compile(
    r"\b(?:\+?1[-.\s]?)?"
    r"(?:\(?\d{3}\)?[-.\s]?)"
    r"\d{3}[-.\s]?\d{4}\b"
)

IP_RE = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)

USERPATH_RE = re.compile(r"/(?:users|home|nfs/users)/[^/\s]+")

def normalize_text(s: str) -> str:

    if not isinstance(s, str):
        return ""

    s = s.replace("\r", "\n")

    s = re.sub(
        r"\n{3,}",
        "\n\n",
        s
    )

    s = re.sub(
        r"[ \t]{2,}",
        " ",
        s
    )

    return s.strip()


def anonymize_text(s: str) -> str:

    s = normalize_text(s)

    s = EMAIL_RE.sub(
        "[EMAIL]",
        s
    )

    s = PHONE_RE.sub(
        "[PHONE]",
        s
    )

    s = IP_RE.sub(
        "[IP]",
        s
    )

    s = USERPATH_RE.sub(
    lambda m: f"/{m.group(0).split('/')[1]}/[USER]",
    s
    )

    return s


# Load RT ticket dataset

print("\nLoading RT ticket dataset...")

df = pd.read_csv(RT_CSV_PATH)

print("DataFrame columns:", df.columns.tolist())
print("Rows:", len(df))


# Clean subject
df["subject_clean"] = (
    df[COL_SUBJECT]
    .fillna("")
    .astype(str)
    .map(anonymize_text)
)


# Clean/anonymize body
df["body_clean"] = (
    df[COL_BODY]
    .fillna("")
    .astype(str)
    .map(anonymize_text)
)


# Combine subject + body
df["problem_text"] = (
    df["subject_clean"]
    + "\n\n"
    + df["body_clean"]
).str.slice(0, 4000)


# Drop empty/very short tickets and exact duplicates
df = df[df["problem_text"].str.strip().str.len() > 20]
df = df.drop_duplicates(subset="problem_text").reset_index(drop=True)

print("Rows after cleaning:", len(df))

#print("\nExample anonymized ticket:")

if len(df) > 0:
    print(df.iloc[0]["body_clean"][:400])

# Load embedding model

print("\nLoading embedding model...")

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Create embeddings

texts = (
    df["problem_text"]
    .fillna("")
    .tolist()
)

print("Creating RT ticket embeddings...")

embs = embedder.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype(np.float32)


# Build FAISS index

index = faiss.IndexFlatIP(
    embs.shape[1]
)

index.add(embs)

print(
    "FAISS index created with",
    index.ntotal,
    "tickets."
)

# RT retrieval

def retrieve_rt(
    query: str,
    top_k: int = TOP_K
):

    q_emb = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    scores, idxs = index.search(
        q_emb,
        top_k
    )

    chunks = []

    for score, idx in zip(
        scores[0],
        idxs[0]
    ):

        # Safety check
        if idx < 0:
            continue

        subject = df.iloc[idx]["subject_clean"]

        text = df.iloc[idx]["problem_text"]

        chunk = (
            f"Source: RT ticket\n"
            #f"Similarity score: {score:.4f}\n"
            f"Subject: {subject}\n"
            f"Text: {text[:CHUNK_CHARS]}"
        )

        chunks.append(chunk)

    return chunks


# Build RAG prompt
def build_rag_prompt(
    query: str,
    retrieved_chunks: list[str]
) -> str:

    context = "\n\n---\n\n".join(
        retrieved_chunks
    )

    prompt = f"""User question:
{query}

Evidence retrieved from RT ticket knowledge base:
{context}

Use the retrieved evidence when relevant.

If the evidence is insufficient, say what information is missing.
"""

    return prompt


# RAG answer

def answer_rag(question: str):

    retrieved_chunks = retrieve_rt(
        question,
        top_k=TOP_K
    )

    prompt = build_rag_prompt(
        question,
        retrieved_chunks
    )

    answer = generate_from_prompt(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS
    )

    return answer, retrieved_chunks


# Interactive chatbot

print(
    "\nMistral 7B + RT Ticket RAG is ready."
)

print(
    "Type your chip/HPC question. "
    "Type 'exit' to quit."
)


while True:

    try:                                          # ← try must be INSIDE while
        question = input("\nAsk a chip/HPC question: ").strip()
    except (EOFError, KeyboardInterrupt):         # ← matches the try above
        print("\nExiting chatbot.")
        break

    if not question:

        print(
            "Please type a question, "
            "or type 'exit' to quit."
        )

        continue


    if question.lower() in [
        "exit",
        "quit"
    ]:

        print(
            "Exiting Mistral Ticket RAG chatbot."
        )

        break


    rag_answer, retrieved_chunks = answer_rag(
        question
    )


    print(
        "\n--- Mistral 7B + RT Ticket RAG Answer ---"
    )

    print(rag_answer)

    # Save QA log

    with open(
        LOG_PATH,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n" + "=" * 80 + "\n"
        )
        f.write(
            f"Question:\n{question}\n\n"
        )
        f.write(
            f"\nAnswer:\n{rag_answer}\n"
        )

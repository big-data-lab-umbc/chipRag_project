import re
import torch
import pandas as pd
import numpy as np
import faiss

from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

RT_CSV_PATH = (
    "/umbc/rs/pi_jianwu/users/tartelt1/CHIPRAG/"
    "allDoitTickets_CLEANED.csv"
)

WIKI_PDF_PATH = (
    "/umbc/rs/pi_jianwu/users/tartelt1/CHIPRAG/"
    "wiki_data.pdf"
)

COL_SUBJECT = "SubjectNoHTML"
COL_BODY = "TransactionContent"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3
RT_CHUNK_CHARS = 1500
WIKI_CHUNK_CHARS = 1200
WIKI_CHUNK_OVERLAP = 200

MAX_NEW_TOKENS = 768

LOG_PATH = "Qwen_wiki+rt_v1.txt"

FALLBACK_MESSAGE = (
    "I could not find enough reliable information in the chip_RT ticket "
    "and Wiki knowledge base to answer this question safely. "
    "Please submit a chip/HPC help ticket so the support team can assist you."
)

# ============================================================
# GPU check
# ============================================================

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    DEVICE = "cuda"
    DTYPE  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  
else:
    print("No GPU found. Model will run on CPU.")
    DEVICE = "cpu"       
    DTYPE  = torch.float32
# ============================================================
# Load Qwen
# ============================================================

print("\nLoading Qwen model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=DTYPE,
    trust_remote_code=True,
)
model.eval()


# ============================================================
# System prompt
# ============================================================
SYSTEM = """You are a support assistant for the UMBC chip HPC cluster.
Use only the retrieved institutional evidence for site-specific facts.
Do not invent commands, paths, partitions, accounts, policies, or system details.
Give safe, step-by-step instructions when the evidence supports them.
If the evidence is insufficient, clearly state what information is missing.
Provide a detailed answer with at least 5 numbered steps.
Explain each step clearly in 2-3 sentences.
Always complete your response fully. Do not stop mid-sentence.
"""

# ============================================================
# Text preprocessing
# ============================================================

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

PATH_RE = re.compile(
    r"(/[\w\-.]+)+"
)


def normalize_text(text: str) -> str:

    if not isinstance(text, str):
        return ""

    text = text.replace("\r", "\n")

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    text = re.sub(
        r"[ \t]{2,}",
        " ",
        text
    )

    return text.strip()


def anonymize_text(text: str) -> str:

    text = normalize_text(text)

    text = EMAIL_RE.sub(
        "[EMAIL]",
        text
    )

    text = PHONE_RE.sub(
        "[PHONE]",
        text
    )

    text = IP_RE.sub(
        "[IP]",
        text
    )

    text = PATH_RE.sub(
        lambda m: re.sub(
            r"/users/[^/]+",
            "/users/[USER]",
            m.group(0)
        ),
        text
    )

    return text


# ============================================================
# Load RT tickets
# ============================================================

print("\nLoading RT tickets...")

df = pd.read_csv(RT_CSV_PATH)

df["subject_clean"] = (
    df[COL_SUBJECT]
    .fillna("")
    .astype(str)
    .map(anonymize_text)
)

df["body_clean"] = (
    df[COL_BODY]
    .fillna("")
    .astype(str)
    .map(anonymize_text)
)

df["problem_text"] = (
    df["subject_clean"]
    + "\n\n"
    + df["body_clean"]
).str.slice(0, 4000)

df = df[
    df["problem_text"].str.strip().str.len() > 20
]

df = df.drop_duplicates(
    subset="problem_text"
).reset_index(drop=True)

print("RT tickets:", len(df))


# ============================================================
# Create combined corpus
# ============================================================

documents = []


# --------------------
# Add RT tickets
# --------------------

for row_number, row in df.iterrows():

    text = normalize_text(
        row["problem_text"]
    )

    if not text:
        continue

    documents.append(
        {
            "source_type": "RT ticket",
            "source_id": f"RT row {row_number}",
            "subject": row["subject_clean"],
            "text": text[:RT_CHUNK_CHARS]
        }
    )


# ============================================================
# Wiki PDF chunking
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = WIKI_CHUNK_CHARS,
    overlap: int = WIKI_CHUNK_OVERLAP
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# Load Wiki PDF
# ============================================================

print("\nLoading Wiki PDF...")

reader = PdfReader(WIKI_PDF_PATH)

print("Wiki PDF pages:", len(reader.pages))


for page_number, page in enumerate(
    reader.pages,
    start=1
):

    text = page.extract_text()

    if not text:
        continue

    text = normalize_text(text)

    page_chunks = chunk_text(text)

    for chunk_number, chunk in enumerate(
        page_chunks,
        start=1
    ):

        documents.append(
            {
                "source_type": "Wiki",
                "source_id": (
                    f"Wiki page {page_number}, "
                    f"chunk {chunk_number}"
                ),
                "subject": "",
                "text": chunk
            }
        )


print("\nCombined knowledge base:")
print("Total chunks:", len(documents))

rt_count = sum(
    1 for d in documents
    if d["source_type"] == "RT ticket"
)

wiki_count = sum(
    1 for d in documents
    if d["source_type"] == "Wiki"
)

print("RT chunks:", rt_count)
print("Wiki chunks:", wiki_count)


if not documents:
    raise ValueError(
        "No documents were loaded."
    )


# ============================================================
# Embedding model
# ============================================================

print("\nLoading embedding model...")

embedder = SentenceTransformer(
    EMBEDDING_MODEL
)

# ============================================================
# Create combined embeddings
# ============================================================

texts = [
    document["text"]
    for document in documents
]


print("Creating combined RT + Wiki embeddings...")

embeddings = embedder.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype(np.float32)

# ============================================================
# Build FAISS index
# ============================================================

index = faiss.IndexFlatIP(
    embeddings.shape[1]
)

index.add(embeddings)


print(
    "FAISS index created with",
    index.ntotal,
    "combined chunks."
)

# ============================================================
# Retrieve from RT + Wiki
# ============================================================

def retrieve_combined(
    query: str,
    top_k: int = TOP_K
):

    query_embedding = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)


    scores, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved = []


    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        document = documents[idx]

        retrieved.append(
            {
                "score": float(score),
                "source_type": document["source_type"],
                "source_id": document["source_id"],
                "subject": document["subject"],
                "text": document["text"]
            }
        )


    return retrieved


# ============================================================
# Build combined RAG prompt
# ============================================================

def build_rag_user_prompt(
    question: str,
    evidence
):

    evidence_parts = []


    for i, item in enumerate(
        evidence,
        start=1
    ):

        part = (
            f"[Evidence {i}]\n"
            f"Source type: {item['source_type']}\n"
            f"Source: {item['source_id']}\n"
        )

        if item["subject"]:
            part += (
                f"Subject: {item['subject']}\n"
            )

        part += (
            f"Text:\n{item['text']}"
        )

        evidence_parts.append(part)


    context = "\n\n---\n\n".join(
        evidence_parts
    )


    return f"""User question:

{question}

Retrieved institutional evidence from chip RT tickets and the chip Wiki:

{context}

Answer the question only if the retrieved evidence supports the answer.

You may combine information from multiple retrieved RT-ticket and Wiki sources
when they are relevant.

Do not invent missing information.

If these sources do not contain enough information to answer safely, output
the fallback test.
"""


# ============================================================
# Qwen generation using chat template
# ============================================================

def generate_answer(
    user_prompt: str
):

    messages = [
        {
            "role": "system",
            "content": SYSTEM
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    # Use apply_chat_template with tokenize=True directly
    # avoids the double tokenization risk
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,                  # ← directly tokenize
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,               # ← returns dict with input_ids
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,              # ← ADD
            pad_token_id=tokenizer.pad_token_id               # ← ADD
                         or tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


# ============================================================
# Complete RAG answer
# ============================================================

def answer_question(
    question: str
):

    retrieved = retrieve_combined(
        question,
        top_k=TOP_K
    )

    if not retrieved:
        prompt = (
            f"User question:\n{question}\n\n"
            "No institutional evidence was retrieved "
            "from the chip RT ticket or Wiki knowledge base. "
            "Answer according to the system instructions."
        )
    else:
        prompt = build_rag_user_prompt(
            question,
            retrieved
        )

    answer = generate_answer(prompt)

    return (
        answer,
        retrieved,
        True
    )


# ============================================================
# Interactive chatbot
# ============================================================

print(
    "\nQwen2.5-3B + RT + Wiki RAG is ready."
)

print(
    "Type your chip/HPC question. "
    "Type 'exit' to quit."
)


while True:

    question = input(
        "\nAsk a chip/HPC question: "
    ).strip()


    if not question:

        print(
            "Please type a question."
        )

        continue


    if question.lower() in [
        "exit",
        "quit"
    ]:

        print(
            "Exiting combined RAG chatbot."
        )

        break


    (
        answer,
        retrieved,
        generation_used
    ) = answer_question(question)


    print(
        "\n--- Qwen2.5-3B + RT + Wiki RAG Answer ---"
    )

    print(answer)


    print(
        "\n--- Retrieval Information ---"
    )

    print(
        "LLM generation used:",
        generation_used
    )


    for i, item in enumerate(
        retrieved,
        start=1
    ):


        print(
            f"\nResult {i}"
        )

        print(
            "Source:",
            item["source_type"]
        )

        print(
            "Location:",
            item["source_id"]
        )

        print(
            item["text"][:500]
        )


    # ========================================================
    # Log results
    # ========================================================

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
            f"\nAnswer:\n{answer}\n"
        )

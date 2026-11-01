import re
import torch
import numpy as np
import faiss

from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

PDF_PATH = (
    "/umbc/rs/pi_jianwu/users/tartelt1/CHIPRAG/"
    "wiki_data.pdf"
)

TOP_K = 3
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
MAX_INPUT_TOKENS = 4096
MAX_NEW_TOKENS = 768

LOG_PATH = "V1_mistral_wiki.txt"


# ============================================================
# GPU check
# ============================================================

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("No GPU found. Model will run on CPU.")


# ============================================================
# Load Mistral model
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available()
    else torch.float32,
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
# Generation
# ============================================================
def generate_from_messages(
    messages: list[dict],
    max_new_tokens: int = MAX_NEW_TOKENS
) -> str:
    inputs = tokenizer.apply_chat_template(   # ← use chat template
        messages,
        tokenize=True,
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
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

# ============================================================
# Text preprocessing
# ============================================================

def normalize_text(text: str) -> str:

    if not isinstance(text, str):
        return ""

    text = text.replace("\r", "\n")

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    # Remove repeated spaces/tabs
    text = re.sub(
        r"[ \t]{2,}",
        " ",
        text
    )

    return text.strip()


# ============================================================
# Read Wiki PDF
# ============================================================

print("\nLoading Wiki PDF...")

reader = PdfReader(PDF_PATH)

print("PDF pages:", len(reader.pages))


pages = []

for page_number, page in enumerate(
    reader.pages,
    start=1
):

    text = page.extract_text()

    if not text:
        continue

    text = normalize_text(text)

    if text:

        pages.append(
            {
                "page": page_number,
                "text": text
            }
        )


print(
    "Pages containing extractable text:",
    len(pages)
)


# ============================================================
# Chunk PDF text
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP
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


wiki_chunks = []


for page_data in pages:

    page_number = page_data["page"]

    text_chunks = chunk_text(
        page_data["text"]
    )

    for chunk_number, chunk in enumerate(
        text_chunks,
        start=1
    ):

        wiki_chunks.append(
            {
                "page": page_number,
                "chunk": chunk_number,
                "text": chunk
            }
        )


print(
    "Total Wiki chunks:",
    len(wiki_chunks)
)


if len(wiki_chunks) == 0:
    raise ValueError(
        "No text chunks were created from the PDF."
    )


print("\nExample Wiki chunk:")
print(wiki_chunks[0]["text"][:500])


# ============================================================
# Load embedding model
# ============================================================

print("\nLoading embedding model...")

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# Create embeddings
# ============================================================

texts = [
    item["text"]
    for item in wiki_chunks
]


print(
    "Creating Wiki embeddings..."
)

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
    "Wiki chunks."
)


# ============================================================
# Retrieve Wiki chunks
# ============================================================

def retrieve_wiki(
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


    retrieved_chunks = []


    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        item = wiki_chunks[idx]

        retrieved_chunks.append(
            {
                "score": float(score),
                "page": item["page"],
                "chunk": item["chunk"],
                "text": item["text"]
            }
        )


    return retrieved_chunks


# ============================================================
# Build Wiki RAG prompt
# ============================================================

def build_rag_prompt(
    query: str,
    retrieved_chunks
):

    evidence_parts = []


    for item in retrieved_chunks:

        evidence_parts.append(
            f"Source: UMBC HPC Wiki PDF\n"
            f"Page: {item['page']}\n"
            f"Text:\n{item['text']}"
        )


    context = "\n\n---\n\n".join(
        evidence_parts
    )


    user_message = (
        f"User question:\n{query}\n\n"
        f"Evidence retrieved from the UMBC HPC Wiki:\n{context}\n\n"
        "Answer using only the retrieved evidence above.\n"
        "If insufficient, state what is missing."
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": user_message},
    ]


# ============================================================
# Wiki RAG answer
# ============================================================

def answer_rag(question: str):

    retrieved_chunks = retrieve_wiki(
        question,
        top_k=TOP_K
    )

    messages = build_rag_prompt(
        question,
        retrieved_chunks
    )

    answer = generate_from_messages(
        messages,
        max_new_tokens=MAX_NEW_TOKENS
    )

    return answer, retrieved_chunks


# ============================================================
# Interactive chatbot
# ============================================================

print(
    "\nMistral 7B + Wiki PDF RAG is ready."
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
            "Exiting Mistral Wiki RAG chatbot."
        )

        break


    wiki_answer, retrieved_chunks = answer_rag(
        question
    )


    print(
        "\n--- Mistral 7B + Wiki RAG Answer ---"
    )

    print(wiki_answer)


    # ========================================================
    # Show retrieved Wiki evidence
    # ========================================================

    print(
        "\n--- Retrieved Wiki Chunks ---"
    )


    for i, item in enumerate(
        retrieved_chunks,
        start=1
    ):

        print(
            f"\nRetrieved Chunk {i}"
        )

        print(
            f"Page: {item['page']}"
        )

        print(
            item["text"]
        )


    # ========================================================
    # Save log
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
            f"\nAnswer:\n{wiki_answer}\n"
        )

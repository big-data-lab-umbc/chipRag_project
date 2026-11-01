import re
from pathlib import Path

import faiss
import numpy as np
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Configuration


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# IMPORTANT: Change this filename to the exact wiki PDF.
PDF_PATH = Path(
    "/umbc/rs/pi_jianwu/users/tartelt1/CHIPRAG/wiki_data.pdf"
)

TOP_K = 3
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MAX_INPUT_TOKENS = 4096
MAX_NEW_TOKENS = 768
LOG_PATH = "Qwen_wiki_v1.txt"


# ============================================================
# GPU and model loading


print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    DEVICE = "cuda"
    DTYPE  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    print("No GPU found. Model will run on CPU.")
    DEVICE = "cpu"       
    DTYPE  = torch.float32

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=DTYPE,
    trust_remote_code=True,
)
model.eval()    # ← ADD THIS
#=============================================================
#System promt
#=============================================================
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
# Text preprocessing and anonymization

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(
    r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"
)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def anonymize_text(text: str) -> str:
    text = normalize_text(text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    #text = PHONE_RE.sub("[PHONE]", text)
    #text = IP_RE.sub("[IP]", text)

    # Anonymize usernames only inside paths containing /users/<username>.
    text = re.sub(r"(/users/)[^/\s]+", r"\1[USER]", text)
    return text


# ============================================================
# Wiki PDF extraction and chunking


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Wiki PDF not found: {pdf_path}\n"
            "Update PDF_PATH so it matches the exact PDF filename."
        )

    reader = PdfReader(str(pdf_path))
    pages: list[dict] = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        clean_text = anonymize_text(raw_text)

        if len(clean_text) < 50:
            print(
                f"Warning: skipping page {page_number}; "
                "it contains little or no extractable text."
            )
            continue

        pages.append(
            {
                "source": pdf_path.name,
                "page": page_number,
                "text": clean_text,
            }
        )

    if not pages:
        raise ValueError(
            "No usable text was extracted from the PDF. "
            "The file may be scanned/image-based and may require OCR."
        )

    return pages


def chunk_pdf_pages(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least 0 and smaller than chunk_size.")

    chunks: list[dict] = []

    for page in pages:
        text = page["text"]
        start = 0
        chunk_number = 1

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if len(chunk_text) >= 100:
                chunks.append(
                    {
                        "source": page["source"],
                        "page": page["page"],
                        "chunk": chunk_number,
                        "text": chunk_text,
                    }
                )

            if end >= len(text):
                break

            start = end - overlap
            chunk_number += 1

    if not chunks:
        raise ValueError("No Wiki chunks were created from the extracted pages.")

    return chunks


print("\nLoading Wiki PDF...")
wiki_pages = extract_pdf_pages(PDF_PATH)
wiki_chunks = chunk_pdf_pages(wiki_pages)
wiki_documents = [chunk["text"] for chunk in wiki_chunks]

print(f"Extracted usable pages: {len(wiki_pages)}")
print(f"Created Wiki chunks: {len(wiki_chunks)}")

print("\nFirst extracted Wiki chunk:")
print("-" * 80)
print(
    f"Source: {wiki_chunks[0]['source']} | "
    f"Page: {wiki_chunks[0]['page']} | "
    f"Chunk: {wiki_chunks[0]['chunk']}"
)
print(wiki_chunks[0]["text"][:1000])
print("-" * 80)


# ============================================================
# Embeddings and FAISS index
# ============================================================

print("\nLoading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

wiki_embeddings = embedding_model.encode(
    wiki_documents,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype(np.float32)

wiki_index = faiss.IndexFlatIP(wiki_embeddings.shape[1])
wiki_index.add(wiki_embeddings)

print(f"FAISS index contains {wiki_index.ntotal} Wiki chunks.")


# ============================================================
# Retrieval and generation
# ============================================================

def retrieve_wiki(question: str, top_k: int = TOP_K) -> list[dict]:
    question = normalize_text(question)
    if not question:
        return []

    effective_top_k = min(top_k, len(wiki_chunks))

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = wiki_index.search(question_embedding, effective_top_k)

    results: list[dict] = []

    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index < 0:
            continue
        chunk = wiki_chunks[int(chunk_index)]
        results.append(
            {
                **chunk,
                "score": float(score),
            }
        )

    return results


def build_wiki_messages(question: str, retrieved: list[dict]) -> list[dict]:
    evidence_sections = []

    for number, result in enumerate(retrieved, start=1):
        evidence_sections.append(
            f"""[Wiki Evidence {number}]
Source: {result["source"]}
PDF page: {result["page"]}
Chunk: {result["chunk"]}
Content:
{result["text"]}"""
        )

    evidence = "\n\n---\n\n".join(evidence_sections)

    user_message = f"""User question:
{question}

Retrieved Wiki evidence:
{evidence}

Answer the question using only relevant evidence above. Cite supporting PDF page numbers.
"""

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message},
    ]


def generate_from_messages(
    messages: list[dict],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()


def answer_wiki_rag(
    question: str,
    top_k: int = TOP_K,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[str, list[dict]]:
    retrieved = retrieve_wiki(question, top_k=top_k)

    if not retrieved:
        return (
            "I could not retrieve usable Wiki evidence for this question.",
            [],
        )

    messages = build_wiki_messages(question, retrieved)
    answer = generate_from_messages(messages, max_new_tokens=max_new_tokens)
    return answer, retrieved


# ============================================================
# Interactive Wiki RAG chatbot
# ============================================================

print("\nQwen 2.5 3B + Wiki PDF RAG is ready.")
print("Type chip/HPC question. Type 'exit' to quit.")

while True:
    try:
        question = input("\nAsk a chip/HPC question: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting Qwen 2.5 3B + Wiki PDF RAG chatbot.")
        break

    if not question:
        print("Please type a question, or type 'exit' to quit.")
        continue

    if question.lower() in {"exit", "quit"}:
        print("Exiting Qwen 2.5 3B + Wiki PDF RAG chatbot.")
        break

    rag_answer, retrieved_results = answer_wiki_rag(question)

    print("\n--- Qwen 2.5 3B + Wiki PDF RAG answer ---")
    print(rag_answer)

    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write(f"Question:\n{question}\n\n")
        log_file.write(f"\nAnswer:\n{rag_answer}\n")

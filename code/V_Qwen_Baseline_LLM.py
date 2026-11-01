import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----- GPU check -----
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    print("No G on CPU (expect ~12 GB RAM and very slow generation).")
    dtype = torch.float32

# ----- Baseline LLM -----
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LOG_FILE = "Baseline_qwen_v1.txt"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    dtype=dtype,              # renamed from torch_dtype
)
model.eval()

SYSTEM = """You are a support assistant for the UMBC chip HPC cluster.
Use only the retrieved institutional evidence for site-specific facts.
Do not invent commands, paths, partitions, accounts, policies, or system details.
Give safe, step-by-step instructions when the evidence supports them.
If the evidence is insufficient, clearly state what information is missing.
Provide a detailed answer with at least 5 numbered steps.
Explain each step clearly in 2-3 sentences.
Always complete your response fully. Do not stop mid-sentence.
No external knowledge base is available in this baseline mode.
"""


def answer_llm(question: str, max_new_tokens: int = 768) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    # Qwen2.5 ChatML format 
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
            do_sample=False,   # deterministic; swap for the sampling block below
            # do_sample=True, temperature=0.4, top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # slice off the prompt at the TOKEN level, not the string level
    generated = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def log_qa(question: str, answer: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Question:\n{question}\n\n")
        f.write(f"Answer:\n{answer}\n")


# ----- Interactive chatbot -----
print("\nQwen 2.5 3B baseline is ready.")
print("Type your chip/HPC question. Type 'exit' to quit.")

while True:
    try:
        question = input("\nAsk a chip/HPC question: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting Qwen 2.5 3B baseline chatbot.")
        break

    if not question:
        continue
    if question.lower() in {"exit", "quit"}:
        print("Exiting Qwen 2.5 3B baseline chatbot.")
        break

    print("\n--- Qwen 2.5 3B ---")
    t0 = time.time()
    answer = answer_llm(question)
    print(answer)
    print(f"\n[{time.time() - t0:.1f}s]")

    log_qa(question, answer)

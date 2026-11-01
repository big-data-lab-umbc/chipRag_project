import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# GPU check
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("No GPU found. Model will run on CPU.")


# Baseline LLM
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
)

model.eval()

# Verify which model was actually loaded
print("Loaded model:", MODEL_NAME)
print("Model config:", model.config._name_or_path)

SYSTEM = """
You are a support assistant for the UMBC chip HPC cluster. 
Use only the retrieved institutional evidence for site-specific facts. 
Do not invent commands, paths, partitions, accounts, policies, or system details.
Give safe, step-by-step instructions when the evidence supports them. If the evidence is insufficient, clearly state what information is missing.
Provide a detailed answer with at least 5 numbered steps.
Explain each step clearly in 2-3 sentences.
Always complete your response fully. Do not stop mid-sentence. No external knowledge base is available in this baseline mode.
"""
def answer_llm(question: str, max_new_tokens: int = 768) -> str:

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
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
            eos_token_id=tokenizer.eos_token_id,     # ← ADD
            pad_token_id=tokenizer.eos_token_id,     # ← ADD
        )

    # Decode only newly generated tokens
    generated_tokens = out[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()

# Interactive chatbot
print("\nMistral 7B baseline is ready.")
print("Type your chip/HPC question. Type 'exit' to quit.")

while True:
    try:                                          # ← try must be INSIDE while
        question = input("\nAsk a chip/HPC question: ").strip()
    except (EOFError, KeyboardInterrupt):         # ← matches the try above
        print("\nExiting chatbot.")
        break

    if question.lower().strip() in ["exit", "quit"]:
        print("Exiting Mistral baseline chatbot.")
        break

    answer = answer_llm(question)

    print("\n--- Mistral 7B No-RAG Answer ---")

    print(answer)

    with open(
        "V1_mistral7b_Baseline.txt",
        "a",
        encoding="utf-8"
    ) as f:

        f.write("\n" + "="*80 + "\n")

        f.write(f"Question:\n{question}\n\n")

        f.write(f"Answer:\n{answer}\n")

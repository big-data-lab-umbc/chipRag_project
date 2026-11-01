## Reproducibility and Experimental Setup

### 1. Computing Environment

The chipRAG experiments were conducted on the University of Maryland, Baltimore County (UMBC) **chip HPC cluster**.

The final experiments used a single compute node with one **NVIDIA L40S GPU**. GPU resources were allocated through the Slurm workload manager.

Both Qwen2.5-3B-Instruct and Mistral-7B-Instruct-v0.3 were executed locally using the Hugging Face Transformers library. Ollama was not used for inference in the final eight experimental configurations.

| Component | Implementation |
|---|---|
| Computing platform | UMBC chip HPC cluster |
| Accelerator | NVIDIA L40S GPU (1 GPU) |
| Scheduler | Slurm |
| Programming language | Python |
| LLM framework | Hugging Face Transformers |
| Models | Qwen2.5-3B-Instruct, Mistral-7B-Instruct-v0.3 |
| Embedding library | Sentence Transformers |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector retrieval | FAISS `IndexFlatIP` |
| Embedding dimension | 384 |
| Retrieval depth | Top-3 chunks |
| Generation strategy | Greedy decoding (`do_sample=False`) |
| Maximum output tokens | 768 |

### 2. Python Environment

The experiments used a Python virtual environment on the HPC cluster.

An example setup based on the development environment is:

```bash
module load Python/3.12.3-GCCcore-13.3.0

python -m venv chiprag_env
source chiprag_env/bin/activate
```

The primary Python dependencies include:

```text
torch
transformers
accelerate
sentence-transformers
faiss-cpu
pandas
numpy
```

Exact package versions should be recorded from the original experiment environment for reproducibility.

### 3. GPU Allocation

Experiments were run using Slurm GPU allocations. An example interactive allocation command is:

```bash
srun --cluster=chip-gpu \
     --partition=gpu \
     --qos=chip-gpu \
     --gres=gpu:1 \
     --mem=24G \
     --time=06:00:00 \
     --pty bash
```

Resource settings and account permissions may need to be adapted to the user's cluster environment. This is an illustrative allocation, not a verified record of every final run.

### 4. Running the Experiments

The study evaluated two models under four configurations:

1. No RAG
2. Historical RT support-ticket RAG
3. Procedural Wiki RAG
4. Combined RT + Wiki RAG

Each configuration was implemented as a separate Python script.

After activating the environment and obtaining GPU resources, execute the corresponding script:

```bash
python -u <experiment_script>.py
```

Replace `<experiment_script>` with the filename of the desired configuration in the `code/` directory.

The scripts may require updates to local input paths, model caches, and output directories before execution on another system.

### 5. Retrieval Configuration

For the RAG-enabled configurations, institutional support content was embedded using `sentence-transformers/all-MiniLM-L6-v2`.

Embeddings were L2-normalized and indexed using FAISS `IndexFlatIP`, enabling cosine-similarity ranking.

The three highest-ranked chunks were retrieved and inserted into the model prompt as supporting context.

For combined RT + Wiki retrieval, both knowledge sources shared one retrieval index, and the top three chunks were selected across the combined corpus.

No minimum similarity-score threshold was applied in the final experiments.

### 6. Response Generation

Both LLMs generated responses using Hugging Face Transformers with deterministic greedy decoding (`do_sample=False`) and a maximum of 768 new tokens.

The prompts instructed the models to provide step-by-step HPC support guidance, avoid unsupported institutional commands or policies, and acknowledge missing information when appropriate.

Each configuration generated responses to the same 25 benchmark HPC support questions.

### 7. Evaluation and Statistical Analysis

Eight LLM configurations and human reference responses were evaluated using seven HPC-specific criteria.

The evaluation involved five HPC experts, 40 completed survey submissions, and 6,982 valid criterion-level ratings.

The statistical analysis included Friedman tests, Wilcoxon signed-rank tests with Holm adjustment, Kendall's W, and Krippendorff's alpha.

### 8. Data Availability and Reproducibility Limitations

The original retrieval corpus includes institutional support tickets and procedural HPC documentation. Public distribution of these sources is subject to institutional data-sharing permissions.

Reproducing the exact responses requires access to the original knowledge sources, model versions, experiment scripts, prompts, and software environment.

Results obtained with alternative support corpora or different hardware/software environments may differ from those reported in the study.


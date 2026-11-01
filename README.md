# Abstract
Large Language Models (LLMs) offer promising capabilities for technical support but often lack institutional knowledge required for reliable high-performance computing (HPC) user assistance. Retrieval-Augmented Generation (RAG) incorporates external knowledge into LLM responses, but its effectiveness may depend on the knowledge source. This study presents chipRAG, an institutional HPC support benchmarking framework investigating how knowledge sources affect expert-perceived response quality. We evaluated two LLMs, namely Qwen2.5-3B and Mistral-7B, under four configurations: no RAG, historical support-ticket RAG, procedural wiki RAG, and combined ticket+wiki RAG. Eight LLM configurations and a human reference were assessed using 25 HPC support questions and seven criteria: relevance, HPC specificity, actionability, correctness, site-policy compliance, hallucination avoidance, and noise-copying avoidance. Five HPC experts completed 40 survey submissions, yielding 6,982 valid criterion-level ratings on a five-point Likert scale. Friedman tests and planned Holm-adjusted Wilcoxon signed-rank comparisons examined differences among configurations and between LLM-generated and human reference responses. Results indicate that wiki-based RAG generally received more favorable ratings on several technical criteria, particularly for Mistral-7B, whereas historical support-ticket RAG showed limited and inconsistent benefits. Combining support tickets and procedural wiki documentation did not consistently outperform individual knowledge sources. Human reference responses received significantly higher ratings than all eight LLM configurations on five of seven evaluation criteria under criterion-level Holm adjustment. Differences in relevance were less consistent, while no significant differences were detected for actionability. These findings highlight the importance of knowledge-source selection and domain-specific expert evaluation when developing RAG-based HPC support systems.

# chipRAG:Benchmarking Large Language Models with Institutional Knowledge-Based Retrieval-Augmented Generation for High-Performance Computing User Support

## Overview

**chipRAG** is a research framework developed at the University of Maryland, Baltimore County (UMBC) to investigate how Retrieval-Augmented Generation (RAG) affects the quality of Large Language Model (LLM) responses for High-Performance Computing (HPC) user support.

The project evaluates whether institutional knowledge retrieved from historical support tickets, procedural Wiki documentation, or both can improve LLM-generated technical support responses.

Two open-source LLMs are evaluated under four retrieval configurations each, resulting in eight LLM systems. Their responses are compared with human-written reference answers using expert evaluations.

This repository is intended to support the transparency, documentation, and reproducibility of the chipRAG experiments.

## Research Questions

**RQ1 — Knowledge-source effects**

How do historical support tickets and procedural Wiki documentation, individually and in combination, affect perceived response quality compared with responses generated without retrieval grounding?

**RQ2 — Human versus LLM responses**

How does the perceived quality of retrieval-grounded LLM responses compare with human-written reference responses across HPC-specific evaluation criteria?

## Models and Experimental Configurations

The following models were evaluated:

- Qwen2.5-3B-Instruct
- Mistral-7B-Instruct-v0.3

Each model was evaluated under four configurations:

| Configuration | Knowledge Source |
|---|---|
| No RAG | Model without retrieval |
| RT RAG | Historical support tickets |
| Wiki RAG | Procedural Wiki documentation |
| RT + Wiki RAG | Combined ticket and Wiki knowledge |

In addition to these eight configurations, human-written reference responses were included as a ninth comparison condition.

## Retrieval Pipeline

The retrieval pipeline includes:

1. Preprocessing institutional knowledge sources.
2. Splitting the retrieved knowledge into text chunks.
3. Generating dense vector embeddings.
4. Indexing embeddings using FAISS.
5. Retrieving the three most similar chunks for each user question.
6. Constructing prompts using the user question, retrieved context, and HPC-specific instructions.
7. Generating responses using locally hosted LLMs.

### Retrieval Settings

| Parameter | Configuration |
|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | FAISS |
| FAISS index | `IndexFlatIP` |
| Embedding dimension | 384 |
| Similarity | Cosine similarity via normalized embeddings |
| Top-k retrieval | 3 |
| Wiki chunk size | Approximately 1,200 characters |
| Wiki chunk overlap | 200 characters |
| RT chunk size | Up to 1,500 characters |
| Similarity threshold | No minimum threshold |
| Inference sampling | `do_sample=False` |
| Maximum new tokens | Up to 768 |

For combined RT + Wiki retrieval, documents from both sources are indexed together, and the top three chunks are selected overall rather than separately from each source.

## Institutional Knowledge Sources

### Historical RT Support Tickets

The original Request Tracker (RT) export contained 17,613 records spanning January 2020 through July 2026.

The retrieval corpus was restricted to tickets from January 2025 onward to reflect the chip HPC environment. After filtering, manual screening, and deduplication, the final corpus contained **486 support tickets**.

### HPC Wiki Documentation

The second knowledge source consists of procedural HPC documentation describing cluster usage, job submission, software environments, and related support procedures.

Wiki documentation was exported and converted into text for indexing and retrieval.

**Data availability:** Institutional support tickets and internal documentation may contain sensitive or restricted information. Thus, raw institutional data should not be publicly distributed.
## Experimental Environment

Experiments were conducted on UMBC's chip HPC cluster using locally hosted language models.

Key technologies:

- Python
- Hugging Face Transformers
- PyTorch
- Sentence Transformers
- FAISS
- Slurm workload manager
- NVIDIA GPU resources

Local inference was used to process institutional support information without transmitting the underlying knowledge corpus to external commercial LLM services.

## Evaluation Methodology

The evaluation included:

| Component | Description |
|---|---|
| Benchmark questions | 25 HPC support questions |
| LLM configurations | 8 |
| Human reference | 1 |
| Total comparison systems | 9 |
| HPC expert evaluators | 5 |
| Completed survey submissions | 40 |
| Valid criterion ratings | 6,982 |
| Rating scale | 1–5 Likert-type scale |

The questions cover common HPC support scenarios such as job scheduling, GPU requests, storage limitations, software environments, and Slurm troubleshooting.

### Evaluation Criteria

Experts rated each response using seven criteria:

1. **Relevance:** How well the response addresses the question.
2. **HPC Specificity:** Whether the response provides appropriate HPC-specific guidance.
3. **Actionability:** Whether the response provides useful steps or instructions.
4. **Correctness:** Whether the technical information appears accurate.
5. **Site-Policy Compliance:** Whether the response follows institutional HPC policies.
6. **Hallucination Avoidance:** Whether the response avoids unsupported or fabricated information.
7. **Noise Copying Avoidance:** Whether the response avoids irrelevant retrieved details.

Higher scores indicate better perceived response quality.

## Statistical Analysis

The statistical analysis uses question-level mean ratings for each system and evaluation criterion.

The following methods were applied:

- **Friedman test:** Compare retrieval configurations across matched benchmark questions.
- **Wilcoxon signed-rank test:** Evaluate planned paired comparisons between configurations.
- **Holm correction:** Adjust for multiple statistical comparisons.
- **Kendall's W:** Describe effect size for Friedman tests.
Statistical significance was evaluated at α = 0.05.



## Experimental Results

### Response-Quality Ratings by System

The following table reports the mean response-quality ratings, standard deviation (SD), and number of valid ratings (N) for each system across seven evaluation criteria.

| System / Condition | Relevance | HPC Specificity | Actionability | Correctness | Site-Policy Compliance | Hallucination Avoidance | Noise Copying Avoidance |
|---|---|---|---|---|---|---|---|
| Qwen2.5-3B (No RAG) | 3.73 ± 1.02 (109) | 3.28 ± 1.05 (109) | 3.03 ± 1.08 (109) | 2.78 ± 1.00 (100) | 2.72 ± 1.13 (109) | 2.83 ± 1.25 (109) | 3.02 ± 1.13 (109) |
| Qwen2.5-3B + RT | 3.41 ± 1.04 (108) | 3.41 ± 1.00 (108) | 2.90 ± 1.13 (108) | 2.97 ± 1.02 (108) | 3.06 ± 1.15 (108) | 2.72 ± 1.22 (108) | 2.62 ± 1.12 (108) |
| Qwen2.5-3B + Wiki | 3.74 ± 0.91 (106) | 3.58 ± 0.99 (106) | 3.20 ± 1.11 (106) | 3.13 ± 1.03 (102) | 3.18 ± 1.09 (106) | 3.14 ± 1.31 (106) | 3.09 ± 1.15 (106) |
| Qwen2.5-3B + RT+Wiki | 3.65 ± 1.17 (114) | 3.63 ± 1.06 (114) | 3.09 ± 1.14 (114) | 3.27 ± 1.16 (110) | 3.28 ± 1.22 (114) | 3.07 ± 1.29 (114) | 3.16 ± 1.25 (114) |
| Mistral-7B (No RAG) | 3.46 ± 1.11 (110) | 3.42 ± 1.16 (110) | 2.81 ± 1.11 (110) | 2.86 ± 1.10 (110) | 2.95 ± 1.19 (110) | 2.75 ± 1.20 (110) | 2.97 ± 1.13 (110) |
| Mistral-7B + RT | 3.34 ± 0.93 (114) | 3.59 ± 0.90 (114) | 2.96 ± 0.99 (114) | 2.94 ± 0.92 (114) | 3.00 ± 1.06 (114) | 2.89 ± 1.13 (114) | 2.56 ± 0.99 (114) |
| Mistral-7B + Wiki | 3.69 ± 0.96 (113) | 3.85 ± 0.95 (114) | 3.26 ± 1.09 (114) | 3.30 ± 0.98 (114) | 3.48 ± 1.07 (114) | 3.26 ± 1.21 (114) | 3.32 ± 1.13 (114) |
| Mistral-7B + RT+Wiki | 3.64 ± 0.96 (113) | 3.70 ± 0.96 (113) | 3.23 ± 0.94 (113) | 3.14 ± 0.93 (113) | 3.27 ± 1.10 (113) | 3.11 ± 1.16 (113) | 3.06 ± 0.99 (113) |
| **Human Reference** | **3.96 ± 1.06 (112)** | **4.17 ± 1.04 (112)** | 3.19 ± 1.31 (112) | **4.12 ± 1.14 (112)** | **4.12 ± 1.08 (112)** | **4.25 ± 1.04 (112)** | **4.19 ± 1.13 (112)** |

**Note:** Values are presented as mean ± standard deviation (N valid ratings). Ratings range from 1 to 5, with higher scores indicating better perceived response quality. Hallucination Avoidance and Noise Copying Avoidance are positively oriented criteria, meaning higher ratings indicate better avoidance. N represents the number of valid ratings, not the number of independent evaluators.



## Summary of Findings

The experiments suggest that retrieval grounding does not consistently improve perceived response quality across all criteria.

- Wiki-grounded retrieval produced favorable results for Mistral-7B on several HPC-specific evaluation criteria.
- Ticket-only retrieval produced limited and inconsistent improvements.
- Combining support tickets and Wiki documentation did not consistently outperform using Wiki documentation alone.
- Human reference responses received significantly higher ratings than the LLM configurations on five technical-reliability criteria after multiple-comparison adjustment.
- Low inter-rater agreement and the institution-specific evaluation setting limit the generalizability of the findings.

These results highlight the importance of knowledge-source selection and domain-specific evaluation when developing RAG systems for institutional technical support.

## Reproducibility

The experimental implementation consists of eight model-configuration scripts, corresponding to the two LLMs and four retrieval conditions.

To reproduce the experiments, check "REPRODUCIBILITY.md", additoionally researchers need access to:

- The corresponding model weights and the Python dependencies used by the implementation find in "REPRODUCIBILITY.md".
- The benchmark support questions check in "Conversation 25.docx.
- Authorized institutional knowledge sources or appropriately prepared substitutes **.
- The experiment scripts and retrieval configuration "code" directory.
- The evaluation and statistical-analysis procedures check "Postprocessing results".

**Repository status:** Experimental scripts, dependency specifications, and approved reproducibility artifacts will be documented as they become available. This README does not imply that restricted institutional datasets are publicly available.

## Limitations

This study uses 25 benchmark questions and expert ratings from one institution. The evaluation measures expert-perceived response quality rather than independently verified technical outcomes. Repeated evaluation by a small expert group and low inter-rater agreement also require cautious interpretation.

## Citation

If you use this project in your research, please cite the associated chipRAG research paper.

*Full paper title, author list, publication venue, and DOI will be added when publication details are finalized.*

## Acknowledgment

This research data was conducted at the University of Maryland, Baltimore County (UMBC) DoIt HPCF. The research, experimental design, code, diagrams, and manuscript were developed by the authors. Anthropic Claude and AI tools accessed through OpenWebUI provided limited assistance with debugging, setting up the Ollama environment, and organizing author-written code. Google Gemini was used to standardize the writing style of human reference responses, as described in the methodology. The authors reviewed the AI-assisted changes and retain full responsibility for the final manuscript and research findings.

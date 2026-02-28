# Sanskrit RAG System

A Retrieval-Augmented Generation (RAG) pipeline for querying Sanskrit documents locally.
Runs entirely on CPU with no API keys, no internet connection, and no GPU required.

---

## Project Structure

```
RAG_Sanskrit_Ansh_Garg/
├── code/
│   ├── document_loader.py      loads and chunks .docx / .txt Sanskrit documents
│   ├── retriever.py            TF-IDF + character n-gram hybrid retriever
│   ├── generator.py            local generation backends (llama.cpp / GPT4All / Extractive)
│   ├── rag_pipeline.py         end-to-end pipeline with command-line interface
│   ├── evaluate.py             retrieval evaluation (Precision@K, MRR, latency)
│   └── requirements.txt        Python dependencies
├── data/
│   └── data.docx               Sanskrit source documents
├── report/
│   └── technical_report.pdf    project report
└── README.md
```

---

## Requirements

- Python 3.8 or higher
- Windows, macOS, or Linux

---

## Step 1 - Install core dependencies

Open a terminal or command prompt, navigate to the project folder, and run:

```
pip install numpy scikit-learn python-docx
```

These three packages are required for all modes of the system.

---

## Step 2 - Choose a generator backend

The system supports three backends. Pick one based on what you need.

### Option A - Extractive (no extra install needed)

Extracts the most relevant sentences directly from retrieved chunks.
No model download, no extra packages. Works immediately after Step 1.

No installation required. Go straight to Step 3.

### Option B - GPT4All (recommended for a full language model)

GPT4All downloads a language model automatically on first run.

```
pip install gpt4all
```

The default model (Phi-3-mini, around 2 GB) will be downloaded the first time you run
the pipeline. Subsequent runs use the cached model.

### Option C - llama-cpp-python (best output quality, requires a model file)

On Windows, use the pre-built wheel to avoid compiler errors:

```
pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Then download a GGUF model file. Two options are listed below.

Small and fast (around 1 GB):
```
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Larger and more accurate (around 4 GB):
```
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

---

## Step 3 - Run the pipeline

All commands below should be run from inside the project folder, the same folder that
contains rag_pipeline.py.

### Basic query using extractive mode

```
python rag_pipeline.py --data_dir data --query "Who is Kalidasa?" --generator extractive
```

### Basic query in Sanskrit

```
python rag_pipeline.py --data_dir data --query "kAlidAsaH kaH?" --generator extractive
```

### Using GPT4All

```
python rag_pipeline.py --data_dir data --query "Who is Kalidasa?" --generator gpt4all
```

### Using llama-cpp-python with a local model file

```
python rag_pipeline.py --data_dir data --query "Who is Kalidasa?" --generator llama_cpp --model_path tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

### Interactive mode

Lets you type multiple queries one by one without restarting the program.

```
python rag_pipeline.py --data_dir data --interactive --generator extractive
```

Type your query and press Enter. Type quit to exit.

### Save the result to a JSON file

```
python rag_pipeline.py --data_dir data --query "Who is Kalidasa?" --generator extractive --output_json result.json
```

### Rebuild the index after adding new documents

```
python rag_pipeline.py --data_dir data --query "Who is Kalidasa?" --generator extractive --force_reindex
```

---

## Step 4 - Run evaluation

```
python evaluate.py --data_dir data --top_k 5
```

This runs six test queries and prints a results table. Results are also saved to
eval_results.json in the same folder.

Metrics explained:

Precision@K    - fraction of the top-K retrieved chunks that are relevant
MRR            - Mean Reciprocal Rank, measures how high the first correct result is ranked
Context Coverage - fraction of query terms found in the retrieved context
Latency (ms)   - wall-clock time per query

---

## Notes on the data directory

The pipeline looks for documents in whatever path you pass to --data_dir.

If your documents are inside the project folder in a subfolder called data:
```
--data_dir data
```

If your documents are one level up from where you are running the script:
```
--data_dir ../data
```

The first time you run the pipeline it builds an index file called sanskrit_index.pkl
and saves it in the current folder. All future runs load this file automatically,
which makes startup faster. If you add or change documents, pass --force_reindex to
rebuild it.

---

## Adding new documents

Place any .docx or .txt file into the data folder, then run:

```
python rag_pipeline.py --data_dir data --query "your query here" --generator extractive --force_reindex
```

---

## System overview

```
User Query
    |
    v
SanskritDocumentLoader
  reads .docx and .txt files
  cleans Unicode text
  splits into overlapping chunks
    |
    v
HybridRetriever
  TF-IDF cosine similarity
  character n-gram overlap bonus
  Devanagari-aware tokeniser
    |
    v
Generator
  ExtractiveGenerator  (built-in, no install)
  GPT4AllGenerator     (pip install gpt4all)
  LlamaCppGenerator    (pip install llama-cpp-python + model file)
    |
    v
Final Answer
```

---

## Observed results

These are the actual results obtained during evaluation on the provided Sanskrit corpus.

Metric               Value
Avg MRR              0.83
Avg Precision@5      0.60
Avg Context Coverage 0.55
Avg Query Latency    1.5 ms

---

## Troubleshooting

0 chunks loaded
The data directory path is wrong or the folder is empty. Make sure your .docx file
is inside the folder you are pointing to with --data_dir. Try --data_dir data if the
data folder is inside the project folder.

empty vocabulary error
Same cause as above. The document was not found or could not be read by the loader.

llama-cpp-python build error on Windows
Use the --prefer-binary install command shown in Option C above. Do not use the
standard pip install llama-cpp-python command on Windows without a C++ compiler.

GPT4All first run is slow
The model downloads once on first run (around 2 GB) and is cached locally.
All subsequent runs start immediately without downloading again.

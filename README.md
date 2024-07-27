# rag-foundation-exercise

## Installation

**Note:** Prefer `python=3.10.*`

### 1. Fork the repo

### 2. Set up environment
```
git clone https://github.com/21522173huy/ai-bootcamp-2024
```

### 3. **Install Required Packages:**

- Install the required packages from `requirements.txt`:

```sh
pip install -r requirements.txt
```

## Homework

### 1. **Fill your implementation**

Search for `"Your code here"` line in the codebase which will lead you to where you should place your code.

### 2. **Run script**
#### Semantic (Question 1)

```
python -m scripts.main \
--mode semantic \
--retrieval_only False \
--force_index True \
--output_path semantic_predictions.jsonl
```
#### Sparse (Question 2)
```
python -m scripts.main \
--mode sparse \
--retrieval_only False \
--force_index True \
--output_path sparse_predictions.jsonl
```
#### NOTE:

To use LLM generation with RAG pipeline, you can use ChatOpenAI by supplying OPENAI_API_KEY in the enviroment variable (supposed you have one).
If you don't have access to OpenAI API, use Groq free-tier instead:

- Register an account at https://console.groq.com/keys (free)
- Generate your API key
- Assign env variable: `export GROQ_API_KEY=<YOUR API KEY>`
- Run the main script without `--retrieval_only` to use LLM

### 3. **Run Evaluation:**
#### Semantic (Question 1)
```
python evaluate_.py \
--predictions semantic_predictions.jsonl \
--gold data/qasper-test-v0.3.json \
--bert_type bert-base-uncased \
--cosine_type sentence-transformers/all-MiniLM-L6-v2
```

#### Sparse (Question 2)
```
!python evaluate_.py \
--predictions sparse_predictions.jsonl \
--gold data/qasper-test-v0.3.json \
--bert_type bert-base-uncased \
--cosine_type sentence-transformers/all-MiniLM-L6-v2
```

### 4. **Results:**
The result code is in [demo.ipynb](demo.ipynb) notebook
#### Semantic (Question 1)
|   | Bert Score | Cosine Similarity |
| -------- | ------- | -------- |
| Answer  |   0.398  |0.155|
| Evidence  |  0.473|   0.281 |

#### Sparse (Question 2)
|   | Bert Score | Cosine Similarity |
| -------- | ------- | -------- |
| Answer  |  0.408 |0.178|
| Evidence  |  0.479|   0.292 |

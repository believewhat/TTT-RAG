# TARSE: Test-Time Adaptation via Retrieval of Skills and Experience for Clinical Reasoning Agents

Code accompanying the ARR submission. The repository's historical name is TTT-RAG.

TARSE separates retrieved **experiences** (reasoning traces from solved cases) from
**skills** (clinical decision rules). Experiences support query-local adaptation;
skills support verification of reasoning steps.

## Repository guide

| File | Purpose |
| --- | --- |
| `train_infer.py` | Legacy query-local LoRA adaptation and candidate reasoning-chain generation. |
| `inference_baseline.py` | Baseline reasoning-chain generation with vLLM. |
| `index_colbert.py` | Index a prepared medical passage collection. |
| `retrieve_knowledge.py` | Retrieve medical passages for prepared queries. |
| `retrieve_math.py` | Retrieve mathematical examples from a prepared collection. |
| `retrieve_hotpotqa.py` | Legacy HotpotQA retrieval utility. |
| `supervised-fine-tune.py` | Supervised fine-tuning utility. |
| `colbert/` | Bundled ColBERT retrieval implementation. |

## Setup and data

The scripts require a CUDA-capable PyTorch environment. Adaptation additionally
uses Transformers, PEFT, and DeepSpeed; baseline generation uses vLLM. Retrieval
uses the bundled ColBERT implementation and its dependencies. Use an environment
compatible with the selected model and CUDA installation.

Model weights, benchmark datasets, prepared reasoning traces, passage collections,
retrieval indexes, and the DeepSpeed configuration are not bundled. Prepare these
artifacts separately and follow their original access and licensing conditions.
Do not include test answers in an adaptation set.

Run scripts from the repository root. Machine-specific locations can be supplied
through these environment variables:

| Variable | Default or requirement |
| --- | --- |
| `TARSE_DEEPSPEED_CONFIG` | Required by `train_infer.py`; path to the experiment's DeepSpeed JSON configuration. |
| `TARSE_BASELINE_INPUT` | `./data/MedQA/test.json` |
| `TARSE_MEDCORP_COLLECTION` | `./data/medcorp_chunks.pickle` |
| `TARSE_WIKIPEDIA_COLLECTION` | `./data/wikipedia/all_chunks.pickle` |
| `TARSE_RETRIEVAL_EXPERIMENT` | `./experiments/notebook` for the legacy retrieval scripts. |

The baseline input is a JSON list containing `id` and `problem`. The adaptation
script expects a JSON list with `input` and `train_finetune`; each retrieved
training item has `input` and `output`. Retrieval scripts expect the prepared TSV
queries, mappings, and collections indicated in their source.

## Reproducibility scope

These are research scripts, not a single end-to-end reproduction command. Model,
prompt, retrieval, and training defaults in individual scripts must be matched to
the particular experiment. The bundled legacy adaptation script is not the later
Qwen3-Base SFT/OPD experiment runner. This release does not add or claim new
benchmark results. The evaluation prompts below are retained from the original
release; their reference answers are for evaluation, not test-time adaptation.

The ARR packaging update refreshes documentation and configurable paths, removes
tracked Python bytecode, and fixes tokenizer initialization order. It does not
constitute a new GPU evaluation or an audit of every legacy training behavior.


## Evaluation Pipeline Prompt Example Template

### 1. Extract Key Points

**Prompt**

Based on the question and answer, summarize ten key points that you consider to be the most crucial from the standard answer. Return the response in the following format: `{1.2.3....}`

Here is the question: `{question}`  
Here is the answer: `{answer}`  

Please do not provide any additional information.

**Output Example (Key Points)**

1. Multifocal electroretinogram (ERG) showed reduced signal in the right eye throughout the macula, confirming the diagnosis of AZOOR.  
2. Acute zonal occult outer retinopathy (AZOOR) was first described by Gass in 1993...  

---

### 2. Extract Diagnostic Reasoning

**Prompt**

Based on the question and answer, please provide a detailed summary of the diagnostic reasoning from the standard answer. Return the response in the following format: `{1.2.3....}`

Here is the question: `{question}`  
Here is the answer: `{answer}`  

Please do not provide any additional information.

**Output Example (Diagnostic Reasoning)**

1. The patient is a 7-year-old boy with a slowly growing, asymptomatic lump on the left lower neck since birth.  
2. Physical examination showed a yellowish, hump-like mass with a hairy surface and cartilage-like consistency near the left sternocleidomastoid muscle...  

---

### 3. Extract Evidence

**Prompt**

Based on the question and answer, please provide a detailed evidence list which is proposed by correct answer. Return the response in the following format: `{1.2.3....}`

Here is the question: `{question}`  
Here is the answer: `{answer}`  

Please do not provide any additional information.

**Output Example (Evidence)**

1. Slowly growing, asymptomatic lump on left lower neck since birth.  
2. Physical examination revealed a yellowish, hump-like mass with hairy surface and cartilage-like consistency.  
3. Ultrasonography indicated a hypoechoic, avascular, bulging nodule with an anechoic tubular structure.  
4. MRI demonstrated a protuberant nodule with diffuse...  

---

### 4. Key Points Score

**Prompt**

Act as a USMLE evaluator, your role involves assessing and comparing a medical student's explanation to the provided target answer. Begin the assessment by carefully reviewing the provided target answer. Then, based on the following specific criteria, determine the score for the student's answer.

Please judge whether the medical student's answer includes these key points (or some other relevant points, but the amount of points must be complete). For example, if the ground truth has 10 key points, and the student answer includes one key point, they will get 0.5 point (if the answer includes 5 points, the score should be 2.5).

Medical student's answer:  
`{answer}`

Key Points:  
`{Key Point}`

Please only return a float number (from 0 to 5). You should check each point one by one (do not judge based on language style such as fluency and so on. Only judge based on whether the student's answer includes correct, relevant, and complete key points). Do not generate any other information.

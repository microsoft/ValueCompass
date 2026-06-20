# CAReDiO: Enhancing Cultural Alignment of LLM via Representativeness and Distinctiveness Guided Data Optimization

CAReDiO is a novel data optimization framework that alternately optimizes culture-sensitive questions and responses according to two information-theoretic objectives in an in-context manner. i) An information gain objective, inspired by Cultural Consensus Theory, to identify samples that better reduce the LLM’s cultural uncertainty and gain agreement, improving representativeness; ii) a culture divergence objective, which is grounded in Cognitive Conflict Theory and theoretically performs a point-wise optimization of different cultures’ JS divergence, to enhance sample distinguishability from non-target cultures, improving distinctiveness. 

---

## Repository layout

```
CAReDiO/
├── 1.data_synthesis.py        # Stage 1: question synthesis + rep/dist optimization
├── 2.data_selection.py        # Stage 2: cluster + score-based sample selection
├── 3.sft_finetune.py          # Stage 3: SFT (with LoRA) on selected data
├── config.py                  # WVS topics, demographic templates, all prompts
├── models.py                  # Unified LLM wrappers (Azure OpenAI / DeepSeek / Qwen / Gemma)
└── scripts/
    ├── 1.data_synthesize.sh
    ├── 2.data_selection.sh
    └── 3.sft_finetune.sh
```

Expected runtime data layout (created automatically under `--output_dir`, e.g. `./data/`):

```
data/
├── WVS_demographics.json                                 # required input: WVS demographic profiles per country
└── <Country>/
    ├── rep_dist_questions_<question_generator>.json      # stage-1 output
    ├── rep_dist_answers_<main_generator>.jsonl          # stage-1 output (per-round)
    ├── embeds/
    │   └── rep_dist_answers_<gen>_round_<r>_embed.jsonl  # stage-2 cache
    ├── figures/
    │   └── <prefix>_<gen>_cluster_threshold_<t>_round_<r>.png
    └── <prefix>_<gen>_cluster_threshold_<t>_round_<r>/
        └── <select_score>_<sample_num>.jsonl             # stage-2 final selection
```

---

## Installation

```bash
# Python 3.12+ recommended (we use 3.12 in the reference `align` conda env)
conda create -n caredio python=3.12 -y
conda activate caredio
pip install -r requirements.txt
```

The pinned `requirements.txt` matches the `align` environment that has been verified to run all three stages.

Provide a `WVS_demographics.json` at `<output_dir>/WVS_demographics.json` keyed by country, with each entry being a list of profile dicts containing `sex / age / education / occupation / social_class / income_level`.

### API authentication

By default `models.py` authenticates to Azure OpenAI / Azure DeepSeek via `AzureCliCredential`, which requires:

```bash
az login
```

**If you do not have `az login` access**, you can swap the auth in two steps without touching the rest of the pipeline. The `__init__` of every Azure-backed wrapper (`create_aoai_client`, `AzureOpenAIModel.__init__`, `AzureDeepSeekModel.__init__`) is the only place that needs to change.

Replace the `AzureCliCredential` + `get_bearer_token_provider` block with a plain API-key client. For example:

```python
# models.py — replacement for the __init__ blocks
import os
from openai import OpenAI, AzureOpenAI

def create_aoai_client():
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-12-01-preview",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        max_retries=5,
    )

class AzureOpenAIModel(LanguageModel):
    def __init__(self, model_name: str, args: dict):
        super().__init__(model_name, args)
        self.aoiclient = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2025-01-01-preview",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            max_retries=5,
        )

class AzureDeepSeekModel(LanguageModel):
    def __init__(self, model_name: str, args: dict):
        super().__init__(model_name, args)
        self.aoiclient = AzureOpenAI(
            api_key=os.environ["AZURE_DEEPSEEK_API_KEY"],
            api_version="2024-12-01-preview",
            azure_endpoint=os.environ["AZURE_DEEPSEEK_ENDPOINT"],
            max_retries=5,
        )
```

Then export the corresponding env vars before running:

```bash
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
export AZURE_DEEPSEEK_API_KEY=...
export AZURE_DEEPSEEK_ENDPOINT="https://<your-deepseek-resource>.services.ai.azure.com"
```

If you prefer the public OpenAI API instead of Azure, replace `AzureOpenAI(...)` with `OpenAI(api_key=os.environ["OPENAI_API_KEY"])` and use the matching public model names in `MODELS` (e.g. `gpt-4o`, `text-embedding-3-small`). The local `QwenOpenModel` / `GemmaModel` wrappers do not need any credentials.

---

## Usage

### Stage 1 — Data synthesis

`scripts/1.data_synthesize.sh` runs two sub-stages of `1.data_synthesis.py`:

- `--stage 1`: generate diverse, deduplicated value-laden questions per topic for a target country.
- `--stage 2`: iterative *representativeness / distinctiveness* optimization that produces multi-perspective answers (demographic profiles + cultural experts + foreign experts), clusters cultural points, scores consensus & distinctiveness, aggregates an integrated answer, and refines the question.

```bash
bash scripts/1.data_synthesize.sh
```

Key flags:

| Flag | Meaning |
| --- | --- |
| `--country` | Target culture, e.g. `Japan` (underscores allowed) |
| `--compare_countries` | Foreign countries used to estimate distinctiveness |
| `--question_generator` / `--main_generator` | Model keys from `models.MODELS` |
| `--batch_size` | Questions optimized per batch |
| `--output_dir` | Root output directory |

### Stage 2 — Data selection

```bash
bash scripts/2.data_selection.sh
```

- Embeds the integrated answers with `text-embedding-3-small`.
- Clusters answers (HAC, cosine, threshold = `--cluster_threshold`).
- Computes per-cluster *representativeness* and *distinctiveness* scores.
- Writes the top-`--sample_num` examples to a JSONL.

### Stage 3 — Supervised fine-tuning

```bash
bash scripts/3.sft_finetune.sh
```

- LoRA SFT via `trl.SFTTrainer`.
- `--align_method "Role-Play"` adds a country-specific system prompt; `"None"` skips it.
- `--train_data_type` supports `plain` and `completion` (see Bugs below for `constant`).

---

## Supported models

Defined in `models.MODELS`. Each entry maps a key (e.g. `openai/gpt-5.2`) to a wrapper class (`AzureOpenAIModel`, `AzureDeepSeekModel`, `QwenOpenModel`, `GemmaModel`). All wrappers expose:

- `get_top_p_answer(prompt_base, prompt_system, max_tokens, temperature, top_p)`
- `get_batch_top_p_answer(...)`
- `get_message_answer(messages, ...)` / `get_batch_message_answer(...)`

Add new models by extending `MODELS` and implementing a wrapper class in `models.py`.

---

## Citation

If you use this code, please cite the CAReDiO paper (add citation here).

# Challenge 1B — Offline PDF Mining, Ranking & Summarization (CPU)

> Extract **structured sections** from collections of PDFs and output a compact JSON with (1) **Top‑N sections** and (2) **short refined snippets** most relevant to a **persona** and **job‑to‑be‑done**.
> Runs **fully offline** inside Docker on **CPU**, with strict thread caps for predictable speed.

---

## 🚀 Docker‑First Quickstart

### 1) Build the image

```bash
# From the repo root (where the Dockerfile and code live)
docker build -t pdf-processor-1b:latest .
```

### 2) Place inputs & (optional) local models

```
.
├─ Collection 1/
│  ├─ challenge1b_input.json
│  └─ PDFs/
│     ├─ file1.pdf
│     └─ file2.pdf
├─ Collection 2/
│  ├─ challenge1b_input.json
│  └─ PDFs/...
└─ models/
   ├─ publaynet_d0_min.pth.tar       # (optional) PubLayNet EfficientDet‑D0
   ├─ msmarco-miniLM/                # (optional) CrossEncoder folder
   └─ all-MiniLM-L6-v2/              # (optional) SentenceTransformer
```

### 3) Run (Windows PowerShell)

```powershell
docker run --rm `
  -v "${PWD}:/app" `
  --env WORKERS=8 `
  --env SKIP_LP_INIT=0 `
  --env USE_CE_RERANK=1 `
  --env EXCLUDE_CAP=60 `
  --network none `
  pdf-processor-1b:latest
```

### 3’) Run (Linux/macOS bash)

```bash
docker run --rm \
  -v "$PWD:/app" \
  -e WORKERS=8 \
  -e SKIP_LP_INIT=0 \
  -e USE_CE_RERANK=1 \
  -e EXCLUDE_CAP=60 \
  --network none \
  pdf-processor-1b:latest
```

**Outputs** are written **in place** under each collection folder:

```
Collection X/challenge1b_output.json
```

---

## ✨ What You Get

* **Parallel PDF extraction** with `ProcessPoolExecutor`. Use `WORKERS=<cores>`.
* **Robust section detection**

  * **PubLayNet** (EfficientDet‑D0 via LayoutParser) if `models/publaynet_d0_min.pth.tar` exists and `SKIP_LP_INIT=0`.
  * **Heuristic fallback** (no weights needed): titles by **font size + UPPERCASE ratio + length**, body by line accumulation.
* **Data‑driven constraints (offline)**

  * **YAKE** → **include** keywords for the persona+task.
  * **WordNet** expansions (synonyms/hyponyms/antonyms) → **includes/excludes/hints**.
  * **`EXCLUDE_CAP`** prevents huge exclude lists from tanking speed/quality.
* **Hybrid ranking**

  * TF‑IDF + page/length priors + lexical coverage + **constraint gating**.
  * Optional **CrossEncoder** re‑rank when `USE_CE_RERANK=1` and local CE model exists.
  * **MMR** for diverse Top‑K (bug‑fixed to avoid rare `KeyError`).
* **Refined snippets**

  * Best short paragraph (\~≤800 chars) per selected section, ending on sentence boundaries.
* **Strict thread caps**

  * Common BLAS/OpenMP threads pinned to 1 to avoid oversubscription.

---

## 🧾 Input JSON (per collection)

`Collection X/challenge1b_input.json`

```json
{
  "persona": { "role": "Travel Planner" },
  "job_to_be_done": { "task": "Plan a trip of 4 days for a group of 10 college friends." },
  "documents": [
    { "filename": "South of France - Cities.pdf" },
    { "filename": "South of France - Beaches.pdf" }
  ]
}
```

---

## ⚙️ Environment Variables

| Variable                 | Default | Purpose                                                                            |
| ------------------------ | ------: | ---------------------------------------------------------------------------------- |
| `WORKERS`                |     `1` | Number of **processes**. Set to **physical cores** (e.g., 8).                      |
| `SKIP_LP_INIT`           |     `0` | `0` = try PubLayNet if weights exist; `1` = **heuristic only**.                    |
| `USE_CE_RERANK`          |     `0` | `1` = use local CrossEncoder to re‑rank top‑M candidates (if model folder exists). |
| `EXCLUDE_CAP`            |    `60` | Hard cap for expanded **excludes** from WordNet.                                   |
| `TOP_K`                  |     `5` | Final number of results (constant in code unless changed).                         |
| `RERANK_TOP_M`           |    `40` | Candidate count for CE re‑rank before blending.                                    |
| `CE_BLEND`               |  `0.45` | Blend weight for CE score into final score.                                        |
| `EMBED_BATCH`            |     `8` | Batch size for SentenceTransformer embeddings.                                     |
| `MAX_SECTIONS_FOR_EMBED` |   `180` | Pre‑selection cap before heavier scoring.                                          |

> The container also sets common CPU thread envs (OMP, MKL, etc.) to **1** to keep each worker single‑threaded for predictable speedups.

---

## 📦 Optional Local Models (offline)

Place **inside your mounted repo** (so the container sees them at `/app/models/...`):

* **PubLayNet EfficientDet‑D0 weights**
  `models/publaynet_d0_min.pth.tar`
* **CrossEncoder** (e.g., MS MARCO MiniLM)
  `models/msmarco-miniLM/` with `config.json`, `pytorch_model.bin`, `tokenizer.*`
* **SentenceTransformer** (e.g., `all-MiniLM-L6-v2`)
  `models/all-MiniLM-L6-v2/` with model + tokenizer files

> The container runs **offline** (`--network none`), so ensure these folders are complete if you enable those features.

---

## 🧭 How It Works (Short)

1. **Discover collections**: scan subfolders for `challenge1b_input.json`.
2. **Extract sections in parallel**: PubLayNet (if available) or heuristic fallback. Normalize text; attach `{source_pdf, page_number}`.
3. **Build constraints** from persona+task: YAKE + WordNet → includes/excludes/hints (capped).
4. **Pre‑select** (optional): TF‑IDF shortlist if many sections.
5. **Score & gate**: TF‑IDF + priors + lexical coverage + include/exclude/hints gates.
6. **Re‑rank** (optional): local CE re‑rank on top‑M and blend score.
7. **Select Top‑K**: MMR for diversity (fixed argmax‑in‑candidate‑set).
8. **Refine snippets**: choose best paragraph per selection (\~≤800 chars).
9. **Write** `challenge1b_output.json` inside each collection.

---

## 🧪 Example Logs

```
--- Running with WORKERS=8 (mode: LP+EffDet, CE_rerank=yes) ---
Processing Collection 1 …
[parallel] Extracting 15 PDFs with 8 workers …
  ↳ 2/15 PDFs done
  ↳ 4/15 PDFs done
  ...
[parallel] Done in 36.5s – 192 sections.
[rank] 192 sections → selecting…
[rank] includes=10 excludes=60 texts=11 (ST=no, CE_rerank=yes)
[write] Collection 1/challenge1b_output.json
```

---

## 🧾 Output Schema

```json
{
  "metadata": {
    "mode": "LP+EffDet | heuristic",
    "workers": 8,
    "timestamp": "2025-07-28T18:40:13Z"
  },
  "extracted_sections": [
    {
      "document": "South of France - Cities.pdf",
      "section_title": "Top Cities for Group Trips",
      "importance_rank": 1,
      "page_number": 3
    }
  ],
  "subsection_analysis": [
    {
      "document": "South of France - Cities.pdf",
      "refined_text": "Nice offers connected beach promenades, group‑friendly hostels...",
      "page_number": 3
    }
  ]
}
```

---

## ⚡ Performance Tips

* Set `WORKERS =` **physical cores** (e.g., 8).
* PubLayNet improves **title quality** but adds compute; heuristic mode is fastest.
* Very high DPI or very large PDFs increase runtime—tune if needed.

---

## 🛠 Troubleshooting

* **MMR `KeyError`**: fixed (argmax taken over current candidate set each iteration).
* **No CE re‑rank** despite `USE_CE_RERANK=1`: check `models/msmarco-miniLM/` exists with model + tokenizer.
* **PubLayNet not activating**: ensure `SKIP_LP_INIT=0` and `models/publaynet_d0_min.pth.tar` exists.
* **Weird bullets or ligatures**: normalization handles most; extend bullet list if your PDFs are exotic.
* **Slow even with many workers**: confirm thread caps are 1 (container sets them), reduce DPI, or use heuristic mode (`SKIP_LP_INIT=1`).

---

## 📜 License & Attribution

* PubLayNet, LayoutParser, YAKE, NLTK/WordNet, and any model artifacts follow their respective licenses.
* Repository code is MIT (unless noted otherwise).

---

## 🙌 Acknowledgements

Thanks to the PubLayNet & LayoutParser communities, and the YAKE/NLTK teams for enabling practical offline NLP/CV.

---

### One‑liner (PowerShell)

```powershell
docker run --rm `
  -v "${PWD}:/app" `
  --env WORKERS=8 `
  --env SKIP_LP_INIT=0 `
  --env USE_CE_RERANK=1 `
  --env EXCLUDE_CAP=60 `
  --network none `
  pdf-processor-1b:latest
```

### One‑liner (bash)

```bash
docker run --rm -v "$PWD:/app" -e WORKERS=8 -e SKIP_LP_INIT=0 -e USE_CE_RERANK=1 -e EXCLUDE_CAP=60 --network none pdf-processor-1b:latest
```

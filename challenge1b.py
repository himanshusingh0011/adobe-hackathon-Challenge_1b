# challenge1b.py
import os, re, json, time
from pathlib import Path
from datetime import datetime
from typing import List

# ─── Strict offline defaults ───────────────────────────────────────────────────
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Limit BLAS/OpenMP threads so parallel workers don't oversubscribe CPUs
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

# Number of parallel processes to use for PDF extraction (set WORKERS env)
WORKERS = int(os.getenv("WORKERS", str(min(8, (os.cpu_count() or 4)))))

# Optionally skip PubLayNet model init (workers will inherit this)
SKIP_LP_INIT = os.getenv("SKIP_LP_INIT", "0") == "1"

# ─── NLTK (local, offline) ─────────────────────────────────────────────────────
os.environ.setdefault("NLTK_DATA", str(Path(__file__).parent / "nltk_data"))
try:
    import nltk  # noqa: F401
    from nltk.corpus import wordnet as wn
    HAVE_WN = True
except Exception:
    HAVE_WN = False

# ---- Safe allowlist for PyTorch ≥2.6 checkpoint unpickling
import argparse, numpy as np
from torch.serialization import add_safe_globals, safe_globals
try:
    add_safe_globals([argparse.Namespace, np.core.multiarray.scalar])
except Exception:
    pass

import torch, fitz
# layoutparser and sentence_transformers are lazily imported below

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from PIL import Image
import yake
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── YAKE offline keyword extractor ────────────────────────────────────────────
yake_extractor = yake.KeywordExtractor(lan="en", n=2, top=10, features=None)

# ==============================
# Configuration / knobs
# ==============================
DEBUG = False

# Layout
DPI             = 150
CONF_MIN        = 0.40
TITLE_MIN_CHARS = 5
TITLE_BAD_WORDS = {
    "conclusion", "appendix", "references",
    "acknowledgments", "acknowledgements"
}
MODEL_FILE = Path(r"C:\adobe-part1b\models\publaynet_d0_min.pth.tar")

# Ranking
EMBED_BATCH            = 8
MAX_SECTIONS_FOR_EMBED = 180
MAX_BODY_CHARS         = 2000
TOP_K                  = 5

# Cross‑encoder re‑ranking
USE_CE_RERANK = True
CE_DIR        = Path(r"C:\adobe-part1b\models\msmarco-miniLM")
RERANK_TOP_M  = 40
CE_BLEND      = 0.45

# Optional local bi‑encoder
ST_LOCAL_CANDIDATES = [
    Path(r"C:\adobe-part1b\models\all-MiniLM-L6-v2"),
    Path("models/all-MiniLM-L6-v2"),
]

# Utility constants
BULLET_MARKS   = {"•","·","●","▪","▫","■","□","–","—","-",
                  "◦","‣","∙","○","o","","","*"}
OVERVIEW_TERMS = {
    "overview","guide","introduction","intro","basics","essentials",
    "checklist","tips","best","top","summary","fundamentals","quick start"
}
STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","for","with","from",
    "by","at","as","is","are","be","been","being","this","that","these",
    "those","it","its","into","your","you","we","our","they","their",
    "them","us","about","over","under","per","via","how","what","when",
    "where","why","which","who","whom","while","during","will","can",
    "could","should","would","may","might","than","then","there","here",
    "also","not","no","yes","do","does","did","done","but","if","so",
    "such","more","most","least","few","many","much","very","just"
}

# ==============================
# WordNet helpers (purely data‑driven)
# ==============================
def wn_synonyms_hyponyms(label: str, max_terms: int = 80):
    """Get (lowercased) synonyms + hyponyms for `label`."""
    if not HAVE_WN:
        return []
    terms = set()
    for syn in wn.synsets(label):
        for lm in syn.lemmas():
            terms.add(lm.name().replace("_"," ").lower())
        if syn.pos() == wn.NOUN:
            for hyp in syn.closure(lambda s: s.hyponyms()):
                for lm in hyp.lemmas():
                    terms.add(lm.name().replace("_"," ").lower())
        if len(terms) >= max_terms:
            break
    return [t for t in terms if t][:max_terms]

def wn_antonyms_and_their_syns(label: str, max_terms: int = 120):
    """Antonyms of `label` + synonyms (and noun hyponyms) of those antonyms."""
    if not HAVE_WN:
        return []
    out = set()
    for syn in wn.synsets(label):
        for lm in syn.lemmas():
            for ant in lm.antonyms():
                a = ant.name().replace("_"," ").lower()
                out.add(a)
                for asyn in wn.synsets(a):
                    for alm in asyn.lemmas():
                        out.add(alm.name().replace("_"," ").lower())
                    if asyn.pos() == wn.NOUN:
                        for hyp in asyn.closure(lambda s: s.hyponyms()):
                            for alm in hyp.lemmas():
                                out.add(alm.name().replace("_"," ").lower())
        if len(out) >= max_terms:
            break
    return [t for t in out if t][:max_terms]

def wn_gloss_noun_seeds(label: str, max_terms: int = 40):
    """Mine noun tokens from WordNet glosses/examples of `label`."""
    if not HAVE_WN:
        return []
    seeds = set()
    def scan_gloss(g: str):
        for tok in re.findall(r"[a-z][a-z\-]+", g.lower()):
            tok = tok.strip("-")
            if wn.synsets(tok, pos=wn.NOUN):
                seeds.add(tok)
    for syn in wn.synsets(label):
        scan_gloss(syn.definition())
        for ex in syn.examples():
            scan_gloss(ex)
    return list(seeds)[:max_terms]

# ==============================
# Layout model (lazy init)  ── CHANGED
# ==============================
layout_model = None

def maybe_init_layout():
    """Initialize PubLayNet model only when explicitly requested (serial path)."""
    global layout_model
    if layout_model is not None or SKIP_LP_INIT:
        return
    try:
        if MODEL_FILE.exists() and is_valid_publaynet_checkpoint(MODEL_FILE):
            # lazy import to avoid importing in workers
            import layoutparser as lp  # type: ignore
            print(f"Using local LP weights: {MODEL_FILE} ({MODEL_FILE.stat().st_size // (1024*1024)} MB)", flush=True)
            layout_model = lp.AutoLayoutModel(
                "lp://PubLayNet/tf_efficientdet_d0/config",
                model_path=str(MODEL_FILE),
                label_map={0:"Text",1:"Title",2:"List",3:"Table",4:"Figure"},
                extra_config={"device":"cpu"},
            )
            if DEBUG:
                print("LayoutParser initialized (EfficientDet-D0, CPU).", flush=True)
        else:
            if DEBUG:
                print("No valid PubLayNet weights; using heuristic extraction.", flush=True)
            layout_model = None
    except Exception as e:
        print(f"[warn] LP init failed: {e}", flush=True)
        layout_model = None

def is_valid_publaynet_checkpoint(p: Path) -> bool:
    try:
        if not p.exists() or p.stat().st_size < 5_000_000:
            return False
        with safe_globals([argparse.Namespace, np.core.multiarray.scalar]):
            obj = torch.load(str(p), map_location="cpu", weights_only=True)
        if isinstance(obj, dict):
            if any(k in obj for k in ("state_dict","ema_state_dict","model")):
                return True
            if obj and all(torch.is_tensor(v) for v in obj.values()):
                return True
        return False
    except Exception:
        return False

# Keep BLAS threads to one in the main process too
torch.set_num_threads(1)

# ─── Offline model loaders (lazy imports) ──────────────────────────────────────
def load_cross_encoder():
    if not USE_CE_RERANK:
        return None
    try:
        from sentence_transformers import CrossEncoder as _CE  # lazy
        ce = _CE(
            str(CE_DIR), device="cpu",
            cache_folder=str(CE_DIR),
            local_files_only=True
        )
        if DEBUG:
            print(f"[info] Loaded CrossEncoder from {CE_DIR}")
        return ce
    except Exception as e:
        print(f"[warn] CE load failed: {e} → continuing without re‑ranking.", flush=True)
        return None

def load_sentence_transformer_local():
    try:
        from sentence_transformers import SentenceTransformer as _ST  # lazy
    except Exception:
        return None
    last_ex = None
    for p in ST_LOCAL_CANDIDATES:
        try:
            if p.exists():
                st = _ST(str(p), device="cpu")
                try:
                    st.max_seq_length = 256
                except Exception:
                    pass
                if DEBUG:
                    print(f"[info] Loaded SentenceTransformer from {p}")
                return st
        except Exception as e:
            last_ex = e
    if DEBUG and last_ex:
        print(f"[info] No local SentenceTransformer: {last_ex}")
    return None

_ce_model = None
_st_model = None
def get_ce_model():
    global _ce_model
    if _ce_model is None:
        _ce_model = load_cross_encoder()
    return _ce_model

def get_st_model():
    global _st_model
    if _st_model is None:
        _st_model = load_sentence_transformer_local()
    return _st_model

# ---- TF‑IDF helper
def tfidf_sim(query, docs):
    vec = TfidfVectorizer(max_features=25000, ngram_range=(1,2))
    X   = vec.fit_transform(docs + [query])
    sims= linear_kernel(X[-1], X[:-1]).ravel()
    return torch.tensor(sims, dtype=torch.float32)

# ==============================
# Text utilities
# ==============================
WS         = re.compile(r"\s+")
PUNCT_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)
LIGATURES  = {
    "\ufb00":"ff","\ufb01":"fi","\ufb02":"fl","\ufb03":"ffi","\ufb04":"ffl",
    "\u2010":"-","\u2011":"-","\u2012":"-","\u2013":"-","\u2014":"-","\u2212":"-"
}

def norm_space(s):
    return WS.sub(" ", s or "").strip()

def normalize_unicode(s):
    if not s: return s
    for k,v in LIGATURES.items():
        s = s.replace(k, v)
    s = s.replace("\xad","")
    return re.sub(r"(\w)-\s+(\w)", r"\1\2", s)

def looks_like_bullet(line):
    ls = line.lstrip()
    return ((ls and (ls[0] in BULLET_MARKS or ls[:2] in {"- ","– ","— ","o ","* "})) or
            bool(re.match(r"^\s*\(?\d+[\.\)]\s+", line)))

def strip_bullet(line):
    s = line.lstrip()
    while s and (s[0] in BULLET_MARKS or s[:2] in {"- ","– ","— ","o ","* "}):
        s = s[1:] if s[0] in BULLET_MARKS else s[2:]
        s = s.lstrip()
    return norm_space(re.sub(r"^\(?\d+[\.\)]\s+","",s))

def plausible_title(t):
    t2 = norm_space(normalize_unicode(t))
    if len(t2)<TITLE_MIN_CHARS or PUNCT_ONLY.match(t2):
        return False
    if any(m in t2 for m in BULLET_MARKS):
        return False
    if t2.lower() in TITLE_BAD_WORDS:
        return False
    return bool(re.search(r"[A-Za-z]", t2))

def split_paragraphs(text):
    text = norm_space(normalize_unicode(text))
    if not text:
        return []
    lines, paras, cur = text.splitlines(), [], []
    for ln in lines:
        if looks_like_bullet(ln):
            cur.append(strip_bullet(ln))
        else:
            if cur:
                paras.append(norm_space(" ".join(cur)))
                cur=[]
            paras.append(norm_space(ln))
    if cur:
        paras.append(norm_space(" ".join(cur)))
    chunks=[]
    for p in paras:
        chunks += [norm_space(x) for x in re.split(r"\n{2,}",p) if norm_space(x)]
    return [c for c in chunks if len(c)>=30]

# ==============================
# Generic query terms
# ==============================
TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9\-]+")
def extract_query_terms(query):
    q   = norm_space(normalize_unicode(query)).lower()
    raw = [t for t in TOKEN_SPLIT.split(q) if t]
    uni, bi = set(), set()
    for t in raw:
        if len(t)>3 and t not in STOPWORDS:
            uni.add(t)
        if "-" in t:
            bi.add(t); bi.add(t.replace("-"," "))
    for i in range(len(raw)-1):
        a,b = raw[i], raw[i+1]
        if len(a)>2 and len(b)>2:
            bi.add(f"{a} {b}")
    return uni, bi

def term_coverage_score(text, uni, bi):
    if not text:
        return 0.0
    t = f" {norm_space(text).lower()} "
    score = sum(0.06 for p in bi if f" {p} " in t)
    score += sum(0.03 for u in uni if f" {u} " in t)
    return min(score, 0.30)

def overview_bonus(title):
    tl = (title or "").lower()
    return 0.08 if any(w in tl for w in OVERVIEW_TERMS) else 0.0

# ==============================
# Build *dynamic* excludes from input (no hard‑coded lists)
# ==============================
NEGATION_SPANS = [
    r"\bno\s+([^.;:]+)",
    r"\bwithout\s+([^.;:]+)",
    r"\bavoid\s+([^.;:]+)",
    r"\bexclude(?:ing)?\s+([^.;:]+)",
    r"\bnot\s+(?:include|contain|use|using)\s+([^.;:]+)",
    r"\b([A-Za-z\-]+)\s*-\s*free\b",
    r"\b([A-Za-z]+)\s+free\b"
]

def _split_items(span: str):
    items=[]
    for part in re.split(r"\s*;\s*", span):
        for it in re.split(r"\s*,\s*|\s+\band\b\s+|\s+\bor\b\s+", part):
            it = norm_space(it)
            if it:
                items.append(it)
    return items

def _filter_keywords(words):
    out, seen = [], set()
    for w in words:
        k = norm_space(w.lower())
        if not k or all(c in "-_/.,:;" for c in k):
            continue
        toks = [t for t in TOKEN_SPLIT.split(k) if t]
        if not toks:
            continue
        if len(toks)==1 and (len(toks[0])<=2 or toks[0] in STOPWORDS):
            continue
        if all(t in STOPWORDS for t in toks):
            continue
        if k not in seen:
            seen.add(k); out.append(k)
    return out

def derive_excludes_from_text(text_lower: str):
    seeds = set()
    for pat in NEGATION_SPANS:
        for m in re.finditer(pat, text_lower, flags=re.IGNORECASE):
            group = m.group(1) if m.groups() else ""
            if group:
                for it in _split_items(group):
                    it = it.replace("-free","").replace(" free","").strip()
                    if it:
                        seeds.add(it)
    excl = set()
    for s in seeds:
        excl.update(wn_synonyms_hyponyms(s))
        excl.update(wn_antonyms_and_their_syns(s))
    return excl

def derive_excludes_from_includes(includes):
    out = set()
    for inc in includes:
        inc_l = inc.lower().strip()
        out.update(wn_antonyms_and_their_syns(inc_l))
        for seed in wn_gloss_noun_seeds(inc_l):
            out.update(wn_synonyms_hyponyms(seed))
        if "-free" in inc_l:
            root = inc_l.replace("-free","").strip()
            if root:
                out.update(wn_synonyms_hyponyms(root))
        elif inc_l.endswith(" free"):
            root = inc_l[:-5].strip()
            if root:
                out.update(wn_synonyms_hyponyms(root))
    return out

def derive_child_gate(text_lower: str):
    if not HAVE_WN:
        return set()
    child_lex = set(wn_synonyms_hyponyms("child")) \
              | set(wn_synonyms_hyponyms("infant")) \
              | set(wn_synonyms_hyponyms("baby"))
    for term in child_lex:
        if re.search(rf"\b{re.escape(term)}\b", text_lower):
            return child_lex
    return set()

# ==============================
# Task → Constraints (YAKE + WordNet, data‑driven)
# ==============================
def parse_task_to_constraints(persona, task):
    text       = norm_space(f"{persona}. {task}")
    text_lower = text.lower()

    kws     = [kw for kw,_ in yake_extractor.extract_keywords(text)]
    include = _filter_keywords(kws)[:12]

    exc_pats = [
        r"\b(must not|should not|shall not|do not|does not|don['’]t|cannot|can['’]t)\s+([^.;:]+)"
    ]
    hint_pats = [
        r"\b(for|aimed at|designed for|intended for)\s+([^.;:]+)",
        r"\b(style|format|tone)\s*:\s*([^.;:]+)"
    ]
    def collect(pats, txt):
        spans=[]
        for pat in pats:
            for m in re.finditer(pat, txt, flags=re.IGNORECASE):
                g = m.groups()[-1] if m.groups() else ""
                if g: spans.append(g.strip())
        return spans

    exclude, hints = [], []
    for span in collect(exc_pats, text):
        exclude += _split_items(span)
    for span in collect(hint_pats, text):
        hints += _split_items(span)

    exclude = _filter_keywords(exclude)
    hints   = _filter_keywords(hints)

    dyn_ex  = set()
    dyn_ex |= derive_excludes_from_text(text_lower)
    dyn_ex |= derive_excludes_from_includes(include)
    dyn_ex |= derive_child_gate(text_lower)

    exclude = sorted({*exclude, *dyn_ex})
    EXCLUDE_CAP = int(os.getenv("EXCLUDE_CAP", "60"))
    if len(exclude) > EXCLUDE_CAP:
        # Prioritize items that actually appear in the input text
        in_text = [w for w in exclude if re.search(rf"\b{re.escape(w)}\b", text_lower)]
        order = in_text + [w for w in exclude if w not in in_text]
        exclude = order[:EXCLUDE_CAP]
    if DEBUG:
        print(f"[constraints] include={include}")
        print(f"[constraints] exclude={exclude}")
        print(f"[constraints] hints={hints}")

    return {"include": include, "exclude": exclude, "hints": hints, "task_text": text}

# ==============================
# MLConstraintScorer
# ==============================
class MLConstraintScorer:
    def __init__(self, st_model=None, ce_model=None):
        self.st = st_model
        self.ce = ce_model

    def _score_st(self, texts, constraints):
        pos = [f"The content satisfies this requirement: {c}." for c in constraints]
        neg = [f"The content violates this requirement: {c}."  for c in constraints]
        Hpos= self.st.encode(pos, convert_to_tensor=True,
                             normalize_embeddings=True, show_progress_bar=False)
        Hneg= self.st.encode(neg, convert_to_tensor=True,
                             normalize_embeddings=True, show_progress_bar=False)
        T   = self.st.encode(texts, convert_to_tensor=True,
                             batch_size=EMBED_BATCH,
                             normalize_embeddings=True, show_progress_bar=False)
        p, n = (T @ Hpos.T).cpu().numpy(), (T @ Hneg.T).cpu().numpy()
        diff = p - n
        if diff.size:
            mn,mx = diff.min(axis=0,keepdims=True), diff.max(axis=0,keepdims=True)
            diff = (diff - mn)/(mx - mn + 1e-8)*2 - 1
        return {c: torch.tensor(diff[:,i], dtype=torch.float32)
                for i,c in enumerate(constraints)}

    def _score_ce(self, texts, constraints):
        out={}
        if not texts or not constraints:
            return out
        def norm(a):
            return (a - a.min())/(a.max() - a.min() + 1e-8) if a.size and a.max()>a.min() else np.zeros_like(a)
        try:
            from sentence_transformers import CrossEncoder as _CE  # ensure import when needed
        except Exception:
            return out
        for c in constraints:
            ph = f"The content satisfies this requirement: {c}."
            nh = f"The content violates this requirement: {c}."
            p = np.asarray(self.ce.predict([(ph, t[:1000]) for t in texts]))
            n = np.asarray(self.ce.predict([(nh, t[:1000]) for t in texts]))
            out[c] = torch.tensor(norm(p) - norm(n), dtype=torch.float32)
        return out

    def score(self, texts, constraints):
        if not constraints:
            return {}
        if self.st:
            return self._score_st(texts, constraints)
        if self.ce:
            return self._score_ce(texts, constraints)
        if DEBUG:
            print("[info] No scoring model; skipping.")
        return {}

# ==============================
# Extraction functions
# ==============================
def _lp_bbox(b):
    if hasattr(b,"block") and hasattr(b.block,"x_1"):
        return b.block.x_1, b.block.y_1, b.block.x_2, b.block.y_2
    if hasattr(b,"coordinates"):
        return tuple(b.coordinates)
    return b.x_1, b.y_1, b.x_2, b.y_2

def extract_sections_with_layout(pdf_path):
    maybe_init_layout()  # ensure initialized if desired
    doc  = fitz.open(pdf_path)
    out  = []
    scale= 72.0 / DPI
    for pn, pg in enumerate(doc, start=1):
        pix    = pg.get_pixmap(dpi=DPI, alpha=False)
        img    = Image.frombytes("RGB",(pix.width,pix.height),pix.samples)
        layout = layout_model.detect(np.array(img))
        blocks = [b for b in layout
                  if b.type in ("Title","Text")
                  and getattr(b,"score",1.0)>=CONF_MIN]
        blocks.sort(key=lambda bb: (_lp_bbox(bb)[1],_lp_bbox(bb)[0]))
        cur=None
        for b in blocks:
            x1,y1,x2,y2 = (_lp_bbox(b)[0]*scale,
                           _lp_bbox(b)[1]*scale,
                           _lp_bbox(b)[2]*scale,
                           _lp_bbox(b)[3]*scale)
            txt = norm_space(normalize_unicode(pg.get_textbox((x1,y1,x2,y2))))
            if not txt:
                continue
            if b.type=="Title" and plausible_title(txt):
                cur={"section_title":txt,"body":"","page_number":pn}
                out.append(cur)
            elif cur and not re.fullmatch(r"^[•·●▪▫○◦]+$",txt):
                cur["body"]+= ("\n\n"+txt) if cur["body"] else txt
    return out

def extract_sections_heuristic(pdf_path):
    doc = fitz.open(pdf_path)
    out=[]
    for pn, pg in enumerate(doc, start=1):
        pd = pg.get_text("dict") or {}
        lines, sizes=[], []
        for b in pd.get("blocks",[]):
            if b.get("type",0)!=0: continue
            for ln in b.get("lines",[]):
                spans=ln.get("spans",[])
                if not spans: continue
                text="".join(s.get("text","") for s in spans)
                text=norm_space(normalize_unicode(text))
                if not text: continue
                svals=[s.get("size",0) for s in spans if "size" in s]
                avg = float(np.mean(svals)) if svals else 0.0
                x1,y1,_,_ = ln.get("bbox",[0,0,0,0])
                lines.append({"text":text,"avg_size":avg,"x":x1,"y":y1})
                sizes.extend(svals)
        if not lines: continue
        thr = np.percentile(sizes,85) if sizes else 0.0
        lines.sort(key=lambda r:(round(r["y"],1),round(r["x"],1)))
        cur=None
        for r in lines:
            t=r["text"]
            is_short = len(t)<90
            alpha    = [c for c in t if c.isalpha()]
            uc       = (sum(c.isupper() for c in alpha)/len(alpha)) if alpha else 0.0
            looks    = ((r["avg_size"]>=thr*0.98 and is_short) or
                        (is_short and uc>0.6))
            if looks and plausible_title(t):
                cur={"section_title":t,"body":"","page_number":pn}
                out.append(cur)
            elif cur and not re.fullmatch(r"^[•·●▪▫○◦]+$",t):
                cur["body"]+= ("\n\n"+t) if cur["body"] else t
    return out

def extract_sections_page_fallback(pdf_path):
    doc=fitz.open(pdf_path)
    out=[]
    for pn, pg in enumerate(doc, start=1):
        pd   = pg.get_text("dict") or {}
        lines=[]
        for b in pd.get("blocks",[]):
            if b.get("type",0)!=0: continue
            for ln in b.get("lines",[]):
                txt="".join(s.get("text","") for s in ln.get("spans",[]))
                txt=norm_space(normalize_unicode(txt))
                if txt: lines.append(txt)
        if not lines: continue
        title = lines[0]
        if not plausible_title(title):
            title = f"Page {pn} – {lines[0][:80]}"
        body="\n".join(lines[1:]) if len(lines)>1 else ""
        out.append({"section_title":title,"body":body,"page_number":pn})
    return out

def extract_sections(pdf_path):
    if not SKIP_LP_INIT and MODEL_FILE.exists():
        # only try LP path when explicitly allowed
        try:
            return extract_sections_with_layout(pdf_path)
        except Exception:
            pass
    secs = extract_sections_heuristic(pdf_path)
    return secs or extract_sections_page_fallback(pdf_path)

# ==============================
# Preselection
# ==============================
def preselect_sections(sections, persona, task, top_m=MAX_SECTIONS_FOR_EMBED):
    if not sections or len(sections)<=top_m:
        return sections
    q     = norm_space(f"{persona} {task}")
    texts = [f"{s['section_title']} {norm_space(s.get('body',''))[:300]}"
             for s in sections]
    try:
        vec=TfidfVectorizer(max_features=25000, ngram_range=(1,2))
        X  = vec.fit_transform(texts + [q])
        sims = linear_kernel(X[-1],X[:-1]).ravel()
        idxs= sims.argsort()[::-1][:top_m]
        return [sections[i] for i in idxs]
    except Exception:
        return sections

# ==============================
# Diversity (MMR)
# ==============================
def mmr_select(sections, scores, k, diversity=0.55):
    if k <= 0 or not sections:
        return []
    N = len(sections)
    picked, cand, docs = [], set(range(N)), set()

    titles = [s["section_title"] or "" for s in sections]
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X = vec.fit_transform(titles)
    sim_tt = torch.tensor(linear_kernel(X, X), dtype=torch.float32)

    base = scores.clone()

    while cand and len(picked) < k:
        # consider only remaining candidates
        cand_idx = torch.tensor(sorted(cand), dtype=torch.long)

        # small penalty if a candidate comes from a doc already used
        pen = torch.zeros(len(cand_idx))
        for li, j in enumerate(cand_idx.tolist()):
            if sections[j]["source_pdf"] in docs:
                pen[li] = 0.10

        if not picked:
            vals = base[cand_idx] - pen
            sel_local = int(torch.argmax(vals))
            i = int(cand_idx[sel_local])
        else:
            sim_to_sel = sim_tt[cand_idx][:, picked].max(dim=1).values
            mmr = (1 - diversity) * (base[cand_idx] - pen) - diversity * sim_to_sel
            sel_local = int(torch.argmax(mmr))
            i = int(cand_idx[sel_local])

        picked.append(i)
        docs.add(sections[i]["source_pdf"])
        cand.remove(i)

    return picked


# ==============================
# Constraint gating utils
# ==============================
def dynamic_include_gate(scores, min_keep):
    if scores.numel()==0:
        return torch.ones(0, dtype=torch.bool)
    arr = scores.cpu().numpy()
    thr = max(float(np.percentile(arr,30)), float(np.median(arr)-0.10))
    mask= scores>=thr
    if mask.sum().item()<min_keep:
        ti= torch.argsort(scores,descending=True)[:min_keep]
        m = torch.zeros_like(scores,dtype=torch.bool)
        m[ti]=True
        mask=m
    return mask

def dynamic_exclude_gate(viol, max_drop_frac=0.5):
    if viol.numel()==0:
        return torch.ones(0, dtype=torch.bool)
    arr   = viol.cpu().numpy()
    thr   = float(np.percentile(arr,70))
    keep  = viol < thr
    dropc = (~keep).sum().item()
    mx    = int(max_drop_frac * viol.numel())
    if dropc > mx:
        kc = int(np.ceil((1-max_drop_frac)*viol.numel()))
        ti = torch.argsort(viol,descending=False)[:kc]
        m  = torch.zeros_like(viol,dtype=torch.bool)
        m[ti]=True
        keep=m
    return keep

# ==============================
# Ranking
# ==============================
def rank_sections(sections, persona, task, top_k=TOP_K):
    if not sections:
        return [], None

    query = norm_space(f"{persona} {task}")
    uni, bi = extract_query_terms(query)
    C = parse_task_to_constraints(persona, task)
    inc_c, exc_c, hints_c = C["include"], C["exclude"], C["hints"]

    


    pool = [s for s in sections
            if plausible_title(s["section_title"]) and norm_space(s.get("body",""))]
    if not pool:
        pool = sections

    titles = [s["section_title"] for s in pool]
    bodies = [norm_space(s.get("body","")[:MAX_BODY_CHARS]) for s in pool]
    combo  = [f"{titles[i]}\n\n{bodies[i]}" for i in range(len(pool))]

    sims         = tfidf_sim(query, combo)
    pages        = torch.tensor([float(s.get("page_number",9999)) for s in pool])
    page_prior   = -0.01*(pages - pages.min())
    length_prior = torch.tensor([min(len(b)/900.0,1.0) for b in bodies]) * 0.03
    title_adj    = torch.tensor([
        overview_bonus(t)
        -0.15*(t.lower() in TITLE_BAD_WORDS)
        -0.25*any(m in t for m in BULLET_MARKS)
        for t in titles
    ])
    cov          = torch.tensor([
        term_coverage_score(f"{titles[i]} {bodies[i]}", uni, bi)
        for i in range(len(pool))
    ])

    score = sims + page_prior + length_prior + title_adj + cov

    st     = get_st_model()
    ce     = get_ce_model()  # still used for final re-rank, below
    scorer = MLConstraintScorer(st_model=st, ce_model=None)  #

    print(f"[rank] includes={len(inc_c)} excludes={len(exc_c)} "
      f"texts={len(pool)} (ST={'yes' if st else 'no'}, CE_rerank={'yes' if ce else 'no'})",
      flush=True)


    # INCLUDE gating (ML + lexical fallback)
    if inc_c:
        inc_scores = scorer.score(combo, inc_c)
        if inc_scores:
            cols = [inc_scores[c] for c in inc_c if inc_scores.get(c) is not None]
            if cols:
                mean_inc = torch.stack(cols, dim=1).mean(dim=1)
                gm = dynamic_include_gate(mean_inc, min_keep=max(6, top_k*2))
                score = score*gm + (-1e6)*(~gm)
            for c in inc_c:
                s = inc_scores.get(c)
                if s is not None:
                    m = dynamic_include_gate(s, min_keep=max(6,top_k*3))
                    score = score*m + (-1e6)*(~m)
                    score += 0.24 * s.clamp(-0.5,0.5)
        else:
            covr   = np.array([sum(bool(re.search(rf"\b{re.escape(c)}\b",txt.lower())) for c in inc_c)
                               for txt in combo],dtype=np.float32)
            covr_t = torch.tensor(covr)
            m      = dynamic_include_gate(covr_t, min_keep=max(6,top_k*3))
            score  = score*m + (-1e6)*(~m)
            score += 0.22 * covr_t

    # EXCLUDE gating (ML + lexical fallback)
    if exc_c:
        vio  = scorer.score(combo, [f"mentions {c}" for c in exc_c])
        comp = scorer.score(combo, [f"avoids   {c}" for c in exc_c])
        if vio and comp:
            for i,c in enumerate(exc_c):
                v = vio.get(f"mentions {c}")
                a = comp.get(f"avoids   {c}")
                if v is not None and a is not None:
                    diff= v - a
                    m   = dynamic_exclude_gate(diff)
                    score = score*m + (-1e6)*(~m)
                    score -= 0.12 * diff.clamp(-0.5,0.5)
        else:
            counts = np.array([sum(bool(re.search(rf"\b{re.escape(c)}\b",txt.lower())) for c in exc_c)
                               for txt in combo],dtype=np.float32)
            cnt_t   = torch.tensor(counts)
            m       = dynamic_exclude_gate(cnt_t, max_drop_frac=0.7)
            score   = score*m + (-1e6)*(~m)
            score  -= 0.35 * cnt_t.clamp(max=5.0)

    # HINTS as soft
    if hints_c:
        hs = scorer.score(combo, [f"about {h}" for h in hints_c])
        for h in hints_c:
            s = hs.get(f"about {h}")
            if s is not None:
                score += 0.06 * s.clamp(-0.5,0.5)

    # CE re‑rank blend
    if ce:
        want  = ", ".join(inc_c) if inc_c else "requirements"
        avoid = ", ".join(exc_c) if exc_c else ""
        ce_q  = f"Return content that satisfies: {want}."
        if avoid:
            ce_q += f" Avoid: {avoid}."
        ce_q += f" Task: {task}"
        order   = torch.argsort(score,descending=True).tolist()
        top_idx = order[: min(RERANK_TOP_M, len(order))]
        pairs   = [(ce_q, combo[i][:1000]) for i in top_idx]
        try:
            cs = np.asarray(ce.predict(pairs))
            if np.std(cs)>1e-6:
                ns = (cs - cs.min())/(cs.max()-cs.min())
            else:
                ns = np.zeros_like(cs)
            for j,i in enumerate(top_idx):
                score[i] += CE_BLEND * float(ns[j])
        except Exception:
            pass

    # Diversity pick
    k    = min(top_k, score.numel())
    idxs = mmr_select(pool, score, k, diversity=0.55)
    return [(pool[i], float(score[i])) for i in idxs], C

# ==============================
# Paragraph refinement
# ==============================
def choose_refined_paragraph(title, body, query, C, scorer):
    paras = split_paragraphs(body)
    if not paras:
        txt = norm_space(normalize_unicode(body))
        if not txt or txt in BULLET_MARKS or len(txt) < 30:
            return title
        return txt[:800] if len(txt)>800 else txt

    q   = norm_space(f"{query} {title}")
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X   = vec.fit_transform(paras + [q])
    sims= linear_kernel(X[-1], X[:-1]).ravel()
    idxs= sims.argsort()[::-1][: min(10, len(paras))]
    cand= [paras[i] for i in idxs]

    inc, exc, hints = C["include"], C["exclude"], C["hints"]
    total = np.zeros(len(cand), dtype=np.float32)

    if inc:
        sc = scorer.score(cand, inc)
        if sc:
            cols = [sc[c] for c in inc if sc.get(c) is not None]
            if cols:
                mean_inc = torch.stack(cols,dim=1).mean(dim=1).cpu().numpy()
                total += 0.45 * mean_inc
            for c in inc:
                s = sc.get(c)
                if s is not None:
                    total += 0.25 * s.cpu().numpy()
        else:
            counts = np.array([sum(bool(re.search(rf"\b{re.escape(c)}\b",txt.lower())) for c in inc)
                               for txt in cand], dtype=np.float32)
            total += 0.60 * counts

    if exc:
        vio  = scorer.score(cand, [f"mentions {c}" for c in exc])
        comp = scorer.score(cand, [f"avoids   {c}" for c in exc])
        if vio and comp:
            for i,c in enumerate(exc):
                v = vio.get(f"mentions {c}")
                a = comp.get(f"avoids   {c}")
                if v is not None and a is not None:
                    total += -0.45 * (v - a).cpu().numpy()
        else:
            counts = np.array([sum(bool(re.search(rf"\b{re.escape(c)}\b",txt.lower())) for c in exc)
                               for txt in cand], dtype=np.float32)
            total += -0.80 * np.minimum(counts, 3.0)

    if hints:
        hs = scorer.score(cand, [f"about {h}" for h in hints])
        for h in hints:
            s = hs.get(f"about {h}")
            if s is not None:
                total += 0.10 * s.cpu().numpy()

    total += 0.08 * sims[idxs]
    best  = int(np.argmax(total))
    out   = cand[best]
    if len(out)>800:
        out = out[:800].rsplit(". ",1)[0] + "."
    return out

# ==============================
# Build output
# ==============================
def build_output(cfg, ranked_with_constraints):
    ranked, C = ranked_with_constraints
    metadata = {
        "input_documents": [d["filename"] for d in cfg["documents"]],
        "persona": cfg["persona"]["role"],
        "job_to_be_done": cfg["job_to_be_done"]["task"],
        "processing_timestamp": datetime.now().isoformat(),
    }
    query  = f"{metadata['persona']} {metadata['job_to_be_done']}"
    st     = get_st_model()
    scorer = MLConstraintScorer(st_model=st, ce_model=None)  

    extracted, analysis = [], []
    for rank,(sec,_) in enumerate(ranked, start=1):
        title   = sec["section_title"]
        body    = sec.get("body","")
        refined = choose_refined_paragraph(title, body, query, C, scorer)
        extracted.append({
            "document": sec["source_pdf"],
            "section_title": title,
            "importance_rank": rank,
            "page_number": sec["page_number"],
        })
        analysis.append({
            "document": sec["source_pdf"],
            "refined_text": refined,
            "page_number": sec["page_number"],
        })

    return {
        "metadata": metadata,
        "extracted_sections": extracted,
        "subsection_analysis": analysis,
    }

# ==============================
# Parallel worker (per-PDF)
# ==============================
def _worker_extract_one(args):
    """
    Run in a separate process – extract sections for a single PDF quickly
    with the heuristic extractor (LP disabled for speed & lower RAM).
    """
    coll_path_str, doc = args

    # ensure one thread inside the worker
    for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "1"
    try:
        import torch as _t
        _t.set_num_threads(1)
    except Exception:
        pass



    from pathlib import Path as _P
    pdf_file = _P(coll_path_str) / "PDFs" / doc["filename"]

    secs = extract_sections(pdf_file) or extract_sections_page_fallback(pdf_file)
    out  = []
    for s in secs:
        s["source_pdf"]    = doc["filename"]
        s["section_title"] = norm_space(normalize_unicode(s["section_title"]))
        s["body"]          = normalize_unicode(s.get("body",""))
        out.append(s)
    return out

# ==============================
# Orchestration
# ==============================
def extract_all_sections_for_collection(coll_path: Path):
    cfg = json.loads((coll_path/"challenge1b_input.json").read_text(encoding="utf-8"))
    docs = cfg["documents"]

    # Parallel path when beneficial
    if WORKERS > 1 and len(docs) > 1:
        # Prevent workers from initializing LP during import
        print(f"[parallel] Extracting {len(docs)} PDFs with {WORKERS} workers …", flush=True)
        t0 = time.time()
        all_secs: List[dict] = []
        tasks = [(str(coll_path), d) for d in docs]
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(_worker_extract_one, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                secs = fut.result()
                all_secs.extend(secs)
                if i % 2 == 0 or i == len(futs):
                    print(f"  ↳ {i}/{len(futs)} PDFs done", flush=True)
        print(f"[parallel] Done in {time.time()-t0:.1f}s – {len(all_secs)} sections.", flush=True)
        return cfg, all_secs

    # Serial path (optionally uses LP if available)
    if not SKIP_LP_INIT:
        maybe_init_layout()

    all_secs=[]
    for doc in docs:
        pdf_file = coll_path/"PDFs"/doc["filename"]
        if DEBUG: print(f"→ Extracting {pdf_file.name}", flush=True)
        secs     = extract_sections(pdf_file)
        if not secs:
            secs = extract_sections_page_fallback(pdf_file)
        for s in secs:
            s["source_pdf"]    = doc["filename"]
            s["section_title"] = norm_space(normalize_unicode(s["section_title"]))
            s["body"]          = normalize_unicode(s.get("body",""))
        all_secs.extend(secs)
    return cfg, all_secs

def process_collection(coll_path: Path):
    cfg, all_secs = extract_all_sections_for_collection(coll_path)
    print(f"[rank] {len(all_secs)} sections → selecting…", flush=True)
    candidates    = preselect_sections(
        all_secs,
        cfg["persona"]["role"],
        cfg["job_to_be_done"]["task"]
    )
    ranked_with_constraints = rank_sections(
        candidates,
        cfg["persona"]["role"],
        cfg["job_to_be_done"]["task"],
        top_k=TOP_K
    )
    out = build_output(cfg, ranked_with_constraints)
    (coll_path/"challenge1b_output.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"[write] {coll_path/'challenge1b_output.json'}", flush=True)
    if DEBUG:
        chosen = [f"{r[0]['source_pdf']} :: {r[0]['section_title']}"
                  for r in ranked_with_constraints[0]]
        print("Chosen sections:\n  " + "\n  ".join(chosen), flush=True)

if __name__ == "__main__":
    # Windows compatibility (especially when packaged)
    try:
        import multiprocessing as mp
        mp.freeze_support()
    except Exception:
        pass

    base = Path(__file__).parent
    mode = "LP+EffDet" if (not SKIP_LP_INIT and MODEL_FILE.exists()) else "heuristic"
    print(f"--- Running with WORKERS={WORKERS} (mode: {mode}) ---", flush=True)

    folders = [f for f in base.iterdir() if f.is_dir() and (f/"challenge1b_input.json").exists()]
    if not folders:
        print("[warn] No collections found (looking for */challenge1b_input.json).", flush=True)

    for folder in folders:
        print(f"Processing {folder.name} …", flush=True)
        process_collection(folder)
        print(f" → {folder.name} done\n", flush=True)

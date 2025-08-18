import pickle
import numpy as np
import pandas as pd
import streamlit as st


def load_pickle(name):
    with open(name, "rb") as f:
        return pickle.load(f)


#  load artifacts
vectorizer = load_pickle("../models/vectorizer.pkl")
X_items = load_pickle("../models/X_items.pkl")
item_index = load_pickle("../models/item_index.pkl")     # tmdbId -> item_idx
index_item = load_pickle("../models/index_item.pkl")     # item_idx -> tmdbId
user_cb_profiles = load_pickle("../models/user_cb_profiles.pkl")
algo_mf = load_pickle("../models/algo_mf.pkl")
algo_sim = load_pickle("../models/algo_item_item.pkl")
uid_to_index = load_pickle("../models/uid_to_index.pkl")
train_seen = load_pickle("../models/train_seen.pkl")
movies_meta = pd.read_csv("../models/movies_meta.csv")


n_items = X_items.shape[0]
id2title = dict(zip(movies_meta["id"], movies_meta["title"]))
id2poster = dict(zip(movies_meta["id"], movies_meta.get("poster_path", pd.Series([None]*len(movies_meta)))))



def preds_cb(uid, k, exclude=True):
    if uid not in uid_to_index: return []
    u_idx = uid_to_index[uid]
    uvec = user_cb_profiles[u_idx, :]
    if uvec.nnz == 0: return []
    sims = (uvec @ X_items.T).toarray().ravel()
    order = np.argsort(-sims)
    seen = train_seen.get(uid, set()) if exclude else set()
    rec = []
    for iidx in order:
        mid = index_item.get(int(iidx))
        if mid is None:
            continue
        if exclude and mid in seen:
            continue
        rec.append((mid, float(sims[iidx])))
        if len(rec) >= k: break
    return rec


def preds_cf(uid, k, method="mf", exclude=True):
    scores = np.zeros(n_items, dtype=np.float32)
    try:
        if method == "mf":
            for i in range(n_items):
                # algo_mf روی Surprise trainset فیت شده؛ اینجا با uid/iid ایندکس را مستقیم فرض می‌گیریم
                scores[i] = algo_mf.predict(str(uid_to_index.get(uid, -1)), str(i)).est
        elif method == "item-item" and algo_sim is not None:
            for i in range(n_items):
                scores[i] = algo_sim.predict(str(uid_to_index.get(uid, -1)), str(i)).est
        else:
            return []
    except Exception:
        return []
    order = np.argsort(-scores)
    seen = train_seen.get(uid, set()) if exclude else set()
    rec = []
    for iidx in order:
        mid = index_item.get(int(iidx))
        if mid is None: continue
        if exclude and mid in seen: continue
        rec.append((mid, float(scores[iidx])))
        if len(rec) >= k: break
    return rec


def preds_hybrid(uid, k, alpha=0.6, cf_method="mf", exclude=True):
    cb = preds_cb(uid, k=max(k*3, 50), exclude=exclude)
    cf = preds_cf(uid, k=max(k*3, 50), method=cf_method, exclude=exclude)
    cb_scores = {m:s for m,s in cb}
    cf_scores = {m:s for m,s in cf}
    merged = set(cb_scores.keys()) | set(cf_scores.keys())
    out = []
    for mid in merged:
        sc_cb = cb_scores.get(mid, 0.0)
        sc_cf = cf_scores.get(mid, 0.0)
        s = alpha*sc_cf + (1-alpha)*sc_cb
        out.append((mid, s, sc_cb, sc_cf))
    out.sort(key=lambda x: -x[1])
    return out[:k]

def poster_url(p):
    if pd.isna(p) or not p: return None
    if str(p).startswith("http"): return p
    return "https://image.tmdb.org/t/p/w342" + str(p)
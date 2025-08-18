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
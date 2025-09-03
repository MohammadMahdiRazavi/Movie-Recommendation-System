import joblib
import numpy as np
import pandas as pd
import streamlit as st
import json
import os
import requests


# Load artifacts
vectorizer = joblib.load("src/models/vectorizer.pkl")

X_items = joblib.load("src/models/X_items.pkl")

item_index = joblib.load("src/models/item_index.pkl")  # tmdbId -> item_idx

index_item = joblib.load("src/models/index_item.pkl")  # item_idx -> tmdbId

user_cb_profiles = joblib.load("src/models/user_cb_profiles.pkl")

algo_mf = joblib.load("src/models/algo_mf.pkl")

algo_sim = joblib.load("src/models/algo_item_item.pkl")

uid_to_index = joblib.load("src/models/uid_to_index.pkl")

train_seen = joblib.load("src/models/train_seen.pkl")

#Load csvs
movies_meta = pd.read_csv("src/models/movies_meta.csv")

popular_movies = pd.read_csv("src/models/pop_candidates.csv")

n_items = X_items.shape[0]
id2title = dict(zip(movies_meta["id"], movies_meta["title"]))
id2poster = dict(zip(movies_meta["id"], movies_meta.get("poster_path", pd.Series([None]*len(movies_meta)))))
id2genres = dict(zip(movies_meta["id"], movies_meta.get("name_genres", pd.Series([""]*len(movies_meta)))))
id2overview = dict(zip(movies_meta["id"], movies_meta.get("overview", pd.Series([""]*len(movies_meta)))))


unique_genres = sorted(set(
        g.strip() for gs in movies_meta["name_genres"].dropna() for g in str(gs).split(",")
    ))
unique_titles = sorted(set(
    t.strip() for t in movies_meta["title"].dropna()
))



if os.path.exists("src/models/user_prefs.json"):
    with open("src/models/user_prefs.json", "r") as f:
        user_prefs = json.load(f)
else:
    user_prefs = {}



#normalize function
def normalize_scores(score_dict):
    if not score_dict:
        return {}
    vals = np.array(list(score_dict.values()))
    min_v, max_v = vals.min(), vals.max()
    if max_v == min_v:
        return {k: 0.5 for k in score_dict}
    return {k: (v - min_v) / (max_v - min_v) for k,v in score_dict.items()}



# Recommender functions
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
        if mid is None: continue
        if exclude and mid in seen: continue
        rec.append((mid, float(sims[iidx])))
        if len(rec) >= k: break
    return rec


def preds_cf(uid, k, method="mf", exclude=True):
    scores = np.zeros(n_items, dtype=np.float32)
    try:
        if method == "mf":
            for i in range(n_items):
                scores[i] = algo_mf.predict(str(uid_to_index.get(uid, -1)), str(i)).est
        elif method == "item-item":
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


def preds_hybrid(uid, k, alpha=0.5, cf_method="mf", exclude=True):
    cb = preds_cb(uid, k=max(k*3, 50), exclude=exclude)
    cf = preds_cf(uid, k=max(k*3, 50), method=cf_method, exclude=exclude)
    cb_scores = {m:s for m,s in cb}
    cf_scores = {m:s for m,s in cf}

    # normalize
    cb_scores = normalize_scores(cb_scores)
    cf_scores = normalize_scores(cf_scores)

    merged = set(cb_scores.keys()) | set(cf_scores.keys())
    out = []
    for mid in merged:
        sc_cb = cb_scores.get(mid, 0.0)
        sc_cf = cf_scores.get(mid, 0.0)
        s = alpha*sc_cf + (1-alpha)*sc_cb
        out.append((mid, s, sc_cb, sc_cf))
    out.sort(key=lambda x: -x[1])
    return out[:k]

def recommend_popular(k):
    return popular_movies["id"].head(k).tolist()


def poster_url(title):
    if title:
        url = f"http://www.omdbapi.com/?apikey=a2e7ab47&t={title}"
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            poster = data.get("Poster")
            if poster and poster != "N/A":
                return poster

    return None


# UI
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender (Hybrid)")

all_users = sorted(list(uid_to_index.keys()))
uid = st.selectbox("User ID (leave empty for new user)", options=["<new user>"] + list(user_prefs.keys()) + all_users)

col1, col2, col3 = st.columns([1,1,1.5])

with col1:
    method = st.radio("CF method", ["mf"] + ["item-item"], horizontal=True)

with col2:
    k = st.slider("Top-K", 5, 30, 5, 1)
with col3:
    alpha = st.slider("Hybrid α (CF weight)", 0.0, 1.0, 0.5, 0.05)

st.markdown("---")


# Cold Start handling
if uid == "<new user>":
    username = st.text_input("Choose a username")
    fav_genres = st.multiselect("Pick your favorite genres:", options=unique_genres)
    fav_titles = st.multiselect("Pick some favorite movies:", options=unique_titles)

    if st.button("Save Preferences"):
        if username.strip() == "":
            st.error("❌ Please enter a username.")
        else:
            user_prefs[username] = {"genres": fav_genres, "titles": fav_titles}
            with open("src/models/user_prefs.json", "w") as f:
                json.dump(user_prefs, f)
            st.success(f"✅ Preferences saved for {username}! Restart app and select from dropdown.")

    recs = []
    if fav_genres:
        genre_mask = movies_meta["name_genres"].apply(
            lambda g: any(gen in str(g) for gen in fav_genres)
        )
        recs = movies_meta[genre_mask].head(k)["id"].tolist()
    elif fav_titles:
        recs = movies_meta[movies_meta["title"].isin(fav_titles)].head(k)["id"].tolist()

    if not recs:
        recs = recommend_popular(k=k)

    st.subheader("Recommendations for You")
    cc = st.columns(5)
    for i, mid in enumerate(recs):
        with cc[i % 5]:
            st.markdown(f"**{id2title.get(mid, mid)}**")
            pu = poster_url(id2title.get(mid))
            if pu: st.image(pu, use_container_width=True)
            st.caption(id2overview.get(mid, ""))


# Existing users (normal recs)
else:
    if uid in uid_to_index:   # old user
        fav_genres = st.multiselect("Pick your favorite genres:", options=unique_genres)
        fav_titles = st.multiselect("Pick some favorite movies:", options=unique_titles)

        recs = []
        if fav_genres:
            genre_mask = movies_meta["name_genres"].apply(
                lambda g: any(gen in str(g) for gen in fav_genres)
            )
            recs = movies_meta[genre_mask].head(k)["id"].tolist()
            st.subheader("Recommendations for You")
            cc = st.columns(5)
            for i, mid in enumerate(recs):
                with cc[i % 5]:
                    st.markdown(f"**{id2title.get(mid, mid)}**")
                    pu = poster_url(id2title.get(mid))
                    if pu: st.image(pu, use_container_width=True)
                    st.caption(id2overview.get(mid, ""))
        elif fav_titles:
            recs = movies_meta[movies_meta["title"].isin(fav_titles)].head(k)["id"].tolist()
            st.subheader("Recommendations for You")
            cc = st.columns(5)
            for i, mid in enumerate(recs):
                with cc[i % 5]:
                    st.markdown(f"**{id2title.get(mid, mid)}**")
                    pu = poster_url(id2title.get(mid))
                    if pu: st.image(pu, use_container_width=True)
                    st.caption(id2overview.get(mid, ""))
        else:
            hyb = preds_hybrid(uid, k=k, alpha=alpha, cf_method=method, exclude=True)
            st.subheader("Hybrid")
            hc = st.columns(5)
            for i, (mid, score, scb, scf) in enumerate(hyb):
                with hc[i % 5]:
                    st.markdown(f"**{id2title.get(mid, mid)}**")
                    pu = poster_url(id2title.get(mid))
                    if pu: st.image(pu, use_container_width=True)
                    st.caption(id2overview.get(mid, ""))

            cb = preds_cb(uid, k=k, exclude=True)
            st.subheader("Content-Based")
            cc = st.columns(5)
            for i, (mid, score) in enumerate(cb):
                with cc[i % 5]:
                    st.markdown(f"**{id2title.get(mid, mid)}**")
                    pu = poster_url(id2title.get(mid))
                    if pu: st.image(pu, use_container_width=True)
                    st.caption(id2overview.get(mid, ""))

            cf = preds_cf(uid, k=k, method=method, exclude=True)
            st.subheader(f"Collaborative Filtering ({method})")
            fc = st.columns(5)
            for i, (mid, score) in enumerate(cf):
                with fc[i % 5]:
                    st.markdown(f"**{id2title.get(mid, mid)}**")
                    pu = poster_url(id2title.get(mid))
                    if pu: st.image(pu, use_container_width=True)
                    st.caption(id2overview.get(mid, ""))


    elif uid in user_prefs:   # new user from JSON
        fav_genres = st.multiselect("Pick your favorite genres:", options=unique_genres)
        fav_titles = st.multiselect("Pick some favorite movies:", options=unique_titles)

        st.subheader(f"Cold-start Profile: {uid}")
        prefs = user_prefs[uid]

        recs = []
        if fav_genres:
            genre_mask = movies_meta["name_genres"].apply(
                lambda g: any(gen in str(g) for gen in fav_genres)
            )
            recs = movies_meta[genre_mask].head(k)["id"].tolist()
            st.subheader("Recommendations for You")
            cc = st.columns(5)
            for i, mid in enumerate(recs):
                with cc[i % 5]:
                    st.markdown(f"**{id2title.get(mid, mid)}**")
                    pu = poster_url(id2title.get(mid))
                    if pu: st.image(pu, use_container_width=True)
                    st.caption(id2overview.get(mid, ""))
        elif fav_titles:
            recs = movies_meta[movies_meta["title"].isin(fav_titles)].head(k)["id"].tolist()
            st.subheader("Recommendations for You")
            cc = st.columns(5)
            for i, mid in enumerate(recs):
                with cc[i % 5]:
                    st.markdown(f"**{id2title.get(mid, mid)}**")
                    pu = poster_url(id2title.get(mid))
                    if pu: st.image(pu, use_container_width=True)
                    st.caption(id2overview.get(mid, ""))
        # Simple recs: match genres first, else popularity
        else:
            mask = movies_meta["name_genres"].fillna("").apply(lambda g: any(gen in g for gen in prefs["genres"]))
            rec_ids = movies_meta[mask]["id"].tolist()
            if not rec_ids:
                rec_ids = recommend_popular(k)
            else:
                rec_ids = rec_ids[:k]

            cc = st.columns(5)
            for i, mid in enumerate(rec_ids):
                with cc[i % 5]:
                    st.markdown(f"**{id2title.get(mid, mid)}**")
                    pu = poster_url(id2title.get(mid))
                    if pu: st.image(pu, use_container_width=True)
                    st.caption(id2overview.get(mid, ""))
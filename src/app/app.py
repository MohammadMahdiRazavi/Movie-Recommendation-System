import pickle
import numpy as np
import pandas as pd
import streamlit as st
import json
import os


# Load artifacts
with open("C:/Users/victus/Desktop/my Project/Machine Learning/Movie-Recommendation-System/src/models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("C:/Users/victus/Desktop/my Project/Machine Learning/Movie-Recommendation-System/src/models/X_items.pkl", "rb") as f:
    X_items = pickle.load(f)

with open("C:/Users/victus/Desktop/my Project/Machine Learning/Movie-Recommendation-System/src/models/item_index.pkl", "rb") as f:
    item_index = pickle.load(f)  # tmdbId -> item_idx

with open("C:/Users/victus/Desktop/my Project/Machine Learning/Movie-Recommendation-System/src/models/index_item.pkl", "rb") as f:
    index_item = pickle.load(f)  # item_idx -> tmdbId

with open("C:/Users/victus/Desktop/my Project/Machine Learning/Movie-Recommendation-System/src/models/user_cb_profiles.pkl", "rb") as f:
    user_cb_profiles = pickle.load(f)

with open("C:/Users/victus/Desktop/my Project/Machine Learning/Movie-Recommendation-System/src/models/algo_mf.pkl", "rb") as f:
    algo_mf = pickle.load(f)

with open("C:/Users/victus/Desktop/my Project/Machine Learning/Movie-Recommendation-System/src/models/algo_item_item.pkl", "rb") as f:
    algo_sim = pickle.load(f)

with open("C:/Users/victus/Desktop/my Project/Machine Learning/Movie-Recommendation-System/src/models/uid_to_index.pkl", "rb") as f:
    uid_to_index = pickle.load(f)

with open("C:/Users/victus/Desktop/my Project/Machine Learning/Movie-Recommendation-System/src/models/train_seen.pkl", "rb") as f:
    train_seen = pickle.load(f)

movies_meta = pd.read_csv("C:/Users/victus/Desktop/my Project/Machine Learning/Movie-Recommendation-System/src/models/movies_meta.csv")


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


USER_PREF_FILE = "C:/Users/victus/Desktop/my Project/Machine Learning/Movie-Recommendation-System/src/app/user_prefs.json"
if os.path.exists(USER_PREF_FILE):
    with open(USER_PREF_FILE, "r") as f:
        user_prefs = json.load(f)
else:
    user_prefs = {}


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


def recommend_popular(k=10):
    popular_ids = movies_meta["id"].sample(frac=1, random_state=42).tolist()  # Dummy popularity
    return popular_ids[:k]


def poster_url(p):
    if pd.isna(p) or not p: return None
    if str(p).startswith("http"): return p
    return "https://image.tmdb.org/t/p/w342" + str(p)


# UI
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender (Hybrid)")

all_users = sorted(list(uid_to_index.keys()))
uid = st.selectbox("User ID (leave empty for new user)", options=["<new user>"] + list(user_prefs.keys()) + all_users)

col1, col2, col3 = st.columns([1,1,1.5])

with col1:
    method = st.radio("CF method", ["mf"] + (["item-item"] if algo_sim is not None else []), horizontal=True)

with col2:
    k = st.slider("Top-K", 5, 30, 5, 1)
with col3:
    alpha = st.slider("Hybrid α (CF weight)", 0.0, 1.0, 0.3, 0.05)

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
            with open(USER_PREF_FILE, "w") as f:
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
            pu = poster_url(id2poster.get(mid))
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
                    pu = poster_url(id2poster.get(mid))
                    if pu: st.image(pu, use_container_width=True)
                    st.caption(id2overview.get(mid, ""))
        elif fav_titles:
            recs = movies_meta[movies_meta["title"].isin(fav_titles)].head(k)["id"].tolist()
            st.subheader("Recommendations for You")
            cc = st.columns(5)
            for i, mid in enumerate(recs):
                with cc[i % 5]:
                    st.markdown(f"**{id2title.get(mid, mid)}**")
                    pu = poster_url(id2poster.get(mid))
                    if pu: st.image(pu, use_container_width=True)
                    st.caption(id2overview.get(mid, ""))
        else:
            hyb = preds_hybrid(uid, k=k, alpha=alpha, cf_method=method, exclude=True)
            st.subheader("Hybrid")
            hc = st.columns(5)
            for i, (mid, score, scb, scf) in enumerate(hyb):
                with hc[i % 5]:
                    st.markdown(f"**{id2title.get(mid, mid)}**")
                    pu = poster_url(id2poster.get(mid))
                    if pu: st.image(pu, use_container_width=True)
                    st.caption(id2overview.get(mid, ""))

            cb = preds_cb(uid, k=k, exclude=True)
            st.subheader("Content-Based")
            cc = st.columns(5)
            for i, (mid, score) in enumerate(cb):
                with cc[i % 5]:
                    st.markdown(f"**{id2title.get(mid, mid)}**")
                    pu = poster_url(id2poster.get(mid))
                    if pu: st.image(pu, use_container_width=True)

            cf = preds_cf(uid, k=k, method=method, exclude=True)
            st.subheader(f"Collaborative Filtering ({method})")
            fc = st.columns(5)
            for i, (mid, score) in enumerate(cf):
                with fc[i % 5]:
                    st.markdown(f"**{id2title.get(mid, mid)}**")
                    pu = poster_url(id2poster.get(mid))
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
                    pu = poster_url(id2poster.get(mid))
                    if pu: st.image(pu, use_container_width=True)
                    st.caption(id2overview.get(mid, ""))
        elif fav_titles:
            recs = movies_meta[movies_meta["title"].isin(fav_titles)].head(k)["id"].tolist()
            st.subheader("Recommendations for You")
            cc = st.columns(5)
            for i, mid in enumerate(recs):
                with cc[i % 5]:
                    st.markdown(f"**{id2title.get(mid, mid)}**")
                    pu = poster_url(id2poster.get(mid))
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
                    pu = poster_url(id2poster.get(mid))
                    if pu: st.image(pu, use_container_width=True)
                    st.caption(id2overview.get(mid, ""))

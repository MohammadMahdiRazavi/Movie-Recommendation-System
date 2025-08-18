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
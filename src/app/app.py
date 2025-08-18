import pickle
import numpy as np
import pandas as pd
import streamlit as st


def load_pickle(name):
    with open(name, "rb") as f:
        return pickle.load(f)
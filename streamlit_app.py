import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
import platform
import pathlib
import os

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# Model va profiling embeddinglarini yuklash
#@st.cache_resource
def load_data():
    with open("model.pkl", "rb") as f:
        data = pickle.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return data["dataset"], data["embeddings"], model

profiles, profile_embeddings, model = load_data()

st.title("NETWORKING ANALYST")

# Foydalanuvchi input formasi
user_about = st.text_area("ABOUT", "")
user_experience = st.text_area("EXPERIENCE", "")
user_skills = st.text_area("SKILLS", "")
user_interests = st.text_area("HEADLINE", "")

if st.button("DEEP RESEARCH 🔭"):
    user_text = f"{user_about} {user_experience} {user_skills} {user_interests}"
    if not user_text.strip():
        st.warning("Iltimos, barcha maydonlarni to'ldiring!")
    else:
        user_embedding = model.encode(user_text, convert_to_numpy=True)
        cos_scores = np.dot(profile_embeddings, user_embedding) / (
            np.linalg.norm(profile_embeddings, axis=1) * np.linalg.norm(user_embedding)
        )
        top_idx = np.argsort(-cos_scores)[:5]
        st.markdown("### Eng mos profillar:")
        for idx in top_idx:
            st.markdown("---")
            st.markdown(f"**MOSLIK EHTIMOLLIGI:** {cos_scores[idx]:.2f}")
            st.markdown(f"**NAME:** {profiles[idx]['FirstName']} {profiles[idx]['LastName']}")
            st.markdown(f"**ABOUT:** {profiles[idx]['About Me']}")
            st.markdown(f"**EXPERIENCE:** {profiles[idx]['Experience']}")
            st.markdown(f"**SKILLS:** {profiles[idx]['Skills']}")
            st.markdown(f"**HEADLINE:** {profiles[idx]['Headline']}")

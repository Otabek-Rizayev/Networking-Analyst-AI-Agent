import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
import platform
import pathlib

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

st.title("Networking uchun do'st topuvchi AI-Agent")

# Model va profiling embeddinglarini yuklash
@st.cache_resource
def load_data():
    with open("model.pkl", "rb") as f:
        data = pickle.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return data["profiles"], data["embeddings"], model

profiles, profile_embeddings, model = load_data()

st.title("LinkedIn Profil Tavsiya Dasturi")

# Foydalanuvchi input formasi
user_about = st.text_area("About Me", "")
user_experience = st.text_area("Experience", "")
user_skills = st.text_area("Skills", "")
user_interests = st.text_area("Headline", "")

if st.button("Eng mos 5 ta profilni topish"):
    user_text = f"{user_about} {user_experience} {user_skills} {user_interests}"
    if not user_text.strip():
        st.warning("Iltimos, barcha maydonlarni to'ldiring.")
    else:
        user_embedding = model.encode(user_text, convert_to_numpy=True)
        cos_scores = np.dot(profile_embeddings, user_embedding) / (
            np.linalg.norm(profile_embeddings, axis=1) * np.linalg.norm(user_embedding)
        )
        top_idx = np.argsort(-cos_scores)[:5]
        st.markdown("### Eng mos profillar:")
        for idx in top_idx:
            st.markdown("---")
            st.markdown(f"**Name:** {profiles[idx]['FirstName']} {profiles[idx]['LastName']}")
            st.markdown(f"**About:** {profiles[idx]['About Me']}")
            st.markdown(f"**Experience:** {profiles[idx]['Experience']}")
            st.markdown(f"**Skills:** {profiles[idx]['Skills']}")
            st.markdown(f"**Headline:** {profiles[idx]['Headline']}")
            st.markdown(f"**Moslik balli:** {cos_scores[idx]:.2f}")

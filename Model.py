from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pickle

# 1. Datasetni yuklash
dataset = load_dataset("ilsilfverskiold/linkedin_profiles_synthetic", split="train")

# 2. Modelni yuklash
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Foydalanuvchi inputini kiritish (misol uchun)
user_about = "Experienced software engineer with a passion for AI."
user_experience = "5 years at Google as a backend developer"
user_skills = "Python, Machine Learning, NLP"
user_interests = "AI, Deep Learning, Open Source"
user_text = f"{user_about} {user_experience} {user_skills} {user_interests}"

# 4. Foydalanuvchi embeddingi
user_embedding = model.encode(user_text, convert_to_tensor=True)

# 5. Datasetdan har bir profil uchun matnli ifoda va embedding yasash
def profile_to_text(example):
    return f"{example['About Me']} {example['Experience']} {example['Skills']} {example['Headline']}"

profile_texts = [profile_to_text(row) for row in dataset]
profile_embeddings = model.encode(profile_texts, convert_to_tensor=True)

# 6. Cosine similarity hisoblash
cos_scores = util.pytorch_cos_sim(user_embedding, profile_embeddings)[0]
top_results = np.argpartition(-cos_scores, range(5))[:5]

# 7. Natijani chiqarish: to'liq profil ma'lumotlari bilan
print("Eng o‘xshash 5 ta profil:")
for idx in top_results.tolist():
    print("="*40)
    print(f"NAME: {dataset[idx]['FirstName']} {dataset[idx]['LastName']}")
    print(f"Location: {dataset[idx]['Location']}")
    print(f"About: {dataset[idx]['About Me']}")
    print(f"Experience: {dataset[idx]['Experience']}")
    print(f"Skills: {dataset[idx]['Skills']}")
    print(f"Interests: {dataset[idx]['Headline']}")
    print(f"O‘xshashlik balli: {cos_scores[idx]:.2f}")

# 8. Modelni saqlash
def profile_to_text(example):
    return f"{example['About Me']} {example['Experience']} {example['Skills']} {example['Headline']}"

profile_texts = [profile_to_text(row) for row in dataset]
profile_embeddings = model.encode(profile_texts, convert_to_numpy=True)

# Profil ma'lumotlari va embeddinglarni faylga saqlash
with open("profiles.pkl", "wb") as f:
    pickle.dump({"profiles": dataset, "embeddings": profile_embeddings}, f)

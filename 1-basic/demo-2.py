from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cosine, euclidean
import pandas as pd
import os
if os.path.exists("models_cache"):
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "models_cache"
else:
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./../models_cache"

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "The cat is lying on the sofa.",
    "The dog is playing in the yard.",
    "An animal is resting on the bed."
]

embeddings = model.encode(sentences)

# Функция для подсчёта всех метрик
def calculate_metrics(emb1, emb2):
    cos_dist = cosine(emb1, emb2)
    cos_sim = 1 - cos_dist
    euc_dist = euclidean(emb1, emb2)
    return cos_dist, cos_sim, euc_dist

# Составляем данные для таблички
pairs = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 0),
]

results = []
for i, j in pairs:
    cos_dist, cos_sim, euc_dist = calculate_metrics(embeddings[i], embeddings[j])
    results.append({
        "Предложение A": sentences[i],
        "Предложение B": sentences[j],
        "Косинусное расстояние": f"{cos_dist:.4f}",
        "Евклидово расстояние": f"{euc_dist:.4f}",
    })

# Отображение результатов в виде таблички
df = pd.DataFrame(results)
print(df.to_markdown(index=False))
# Стат метрики Косинусного расстояния
print("\nСтатистика Косинусного расстояния:")
print(f"Среднее: {df['Косинусное расстояние'].astype(float).mean():.4f}")
print(f"Максимальное: {df['Косинусное расстояние'].astype(float).max():.4f}")
print(f"Минимальное: {df['Косинусное расстояние'].astype(float).min():.4f}")
# Отношение минимального и максимального значений
print(f"Отношение min/max: {df['Косинусное расстояние'].astype(float).min() / df['Косинусное расстояние'].astype(float).max():.4f}")

# Стат метрики Евклидово расстояние
print("\nСтатистика Евклидова расстояния:")
print(f"Среднее: {df['Евклидово расстояние'].astype(float).mean():.4f}")
print(f"Максимальное: {df['Евклидово расстояние'].astype(float).max():.4f}")
print(f"Минимальное: {df['Евклидово расстояние'].astype(float).min():.4f}")
# Отношение минимального и максимального значений
print(f"Отношение min/max: {df['Евклидово расстояние'].astype(float).min() / df['Евклидово расстояние'].astype(float).max():.4f}")

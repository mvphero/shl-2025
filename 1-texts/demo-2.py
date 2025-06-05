from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cosine, euclidean
import pandas as pd
import os
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
    dot_prod = np.dot(emb1, emb2)
    return cos_dist, cos_sim, euc_dist, dot_prod

# Составляем данные для таблички
pairs = [
    (0, 1),
    (0, 2),
    (1, 2),
    (0, 0),
    (1, 1),
    (2, 2),
    (1, 0),
]

results = []
for i, j in pairs:
    cos_dist, cos_sim, euc_dist, dot_prod = calculate_metrics(embeddings[i], embeddings[j])
    results.append({
        "Предложение A": sentences[i],
        "Предложение B": sentences[j],
        "Косинусное расстояние": f"{cos_dist:.4f}",
        "Евклидово расстояние": f"{euc_dist:.4f}",
    })

# Отображение результатов в виде таблички
df = pd.DataFrame(results)
print(df.to_markdown(index=False))

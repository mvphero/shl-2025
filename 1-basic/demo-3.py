import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
if os.path.exists("models_cache"):
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "models_cache"
else:
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./../models_cache"

# 1. Загружаем данные
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "data", "products.csv")
products = pd.read_csv(csv_path)

# 2. Подготовим тексты для эмбеддингов
texts = (products["name"] + ". " + products["description"]).tolist()

# 3. Создаём эмбеддинги товаров
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)

# 4. Кластеризация эмбеддингов через K-Means
num_clusters = 4  # Можешь менять количество кластеров
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)

products["cluster"] = clusters

# 5. Визуализация результатов с помощью PCA (снижение размерности)
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 6))

for cluster_id in range(num_clusters):
    cluster_points = reduced_embeddings[clusters == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id}")

# подписываем точки товарами
for i, txt in enumerate(products["name"]):
    plt.annotate(txt, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=15, alpha=1, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))

# 6. Выводим примеры товаров по кластерам
for cluster_id in range(num_clusters):
    print(f"\nCluster {cluster_id} products:")
    print(products[products["cluster"] == cluster_id][["name", "description"]].to_markdown(index=False))

plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig("kmeans_clustering-1.png", dpi=300, bbox_inches='tight')
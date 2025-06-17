import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from PIL import Image
import os

if os.path.exists("models_cache"):
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "models_cache"
else:
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./../models_cache"


script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(script_dir, "data/feed_with_images.csv")
IMAGES_DIR = os.path.join(script_dir, "data/product_images")

# подключение к Qdrant
# For Docker on Windows\MAC, use "http://host.docker.internal:6333"
# QDRANT_URL = "http://host.docker.internal:6333"
# For Docker on Linux use
# QDRANT_URL = "http://172.17.0.1:6333"
QDRANT_URL = "http://host.docker.internal:6333"
client = QdrantClient(QDRANT_URL)
COLLECTION_NAME = "products"

text_model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
image_model = SentenceTransformer('clip-ViT-B-32')

# Проверка и создание коллекции с двумя эмбеддингами
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "image": models.VectorParams(size=512, distance=models.Distance.COSINE),
        "text": models.VectorParams(size=512, distance=models.Distance.COSINE),
    }
)

df = pd.read_csv(DATA_CSV)

# Индексация данных
for _, row in df.iterrows():
    image_path = os.path.join(IMAGES_DIR, row['picture'])
    image = Image.open(image_path).convert("RGB")

    combined_text = f"{row['category_name']} {row['name']} {row['description']}"
    text_embedding = text_model.encode(combined_text).tolist()
    img_embedding = image_model.encode(image).tolist()

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=row['ID'],
                vector={
                    "image": img_embedding,
                    "text": text_embedding,
                },
                payload={
                    "name": row['name'],
                    "description": row['description'],
                    "category_name": row['category_name'],
                    "picture": row['picture'],
                }
            )
        ]
    )

print("✅ Индексация обоих эмбеддингов завершена.")
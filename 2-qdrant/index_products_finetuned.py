import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from PIL import Image
import os

os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./../models_cache"

DATA_CSV = "data/feed_with_images.csv"
IMAGES_DIR = "data/product_images"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "fine_tuned_products"

client = QdrantClient(QDRANT_URL)

# Загружаем fine-tuned текстовую модель и исходную модель для изображений
text_model = SentenceTransformer('./fine_tuned_text_model')
image_model = SentenceTransformer('clip-ViT-B-32')

# Проверка и создание коллекции
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "image": models.VectorParams(size=512, distance=models.Distance.COSINE),
        "text": models.VectorParams(size=512, distance=models.Distance.COSINE),
    }
)

# Загружаем данные
df = pd.read_csv(DATA_CSV)

# Индексация данных с fine-tuned моделью
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

print("✅ Индексация с fine-tuned моделью завершена.")

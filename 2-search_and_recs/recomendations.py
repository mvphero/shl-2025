from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from qdrant_client import QdrantClient, models
import random
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# раздача изображений товаров
app.mount("/static", StaticFiles(directory=os.path.abspath("data/product_images")), name="static")


# раздача фронтенда
@app.get("/")
async def get_frontend():
    return FileResponse(os.path.join(BASE_DIR, "frontend/recs.html"))

# подключение к Qdrant
client = QdrantClient("http://localhost:6333")
COLLECTION_NAME = "fine_tuned_products"

@app.get("/random")
async def get_random_products(limit: int = 16):
    """Возвращает случайные товары из коллекции"""
    response, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=limit,
        with_payload=True,
    )
    random.shuffle(response)
    return [{
        "id": rec.id,
        "name": rec.payload.get("name", "No name"),
        "image_url": f"{rec.payload.get('picture')}",
    } for rec in response]

@app.get("/recommend")
async def recommend(viewed_ids: str = Query("", description="Comma separated IDs")):
    viewed_list = viewed_ids.split(",") if viewed_ids else []

    if not viewed_list:
        # Fetch random items if no viewed items provided
        random_response, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,  # increase this if you have many items
            with_payload=True,
        )
        random.shuffle(random_response)
        recommendations = random_response[:8]
    else:
        recommendations = client.recommend(
            collection_name=COLLECTION_NAME,
            positive=viewed_list[-3:],  # last 3 viewed products
            strategy=models.RecommendStrategy.AVERAGE_VECTOR,
            using="image",
            limit=8,
            with_payload=True
        )
        recommendations = [rec for rec in recommendations if rec.id not in viewed_list]

    return [{
        "id": rec.id,
        "name": rec.payload.get("name", "No name"),
        "image_url": f"{rec.payload.get('picture')}",
    } for rec in recommendations]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("recomendations:app", host="0.0.0.0", port=8000, reload=True)

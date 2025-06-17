from fastapi import FastAPI, UploadFile, File, Query
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from PIL import Image
import io
import os
from functools import lru_cache
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


if os.path.exists("models_cache"):
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "models_cache"
else:
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./../models_cache"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COLLECTION_NAME = "products"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "data/product_images")), name="static")
# For Docker on Windows\MAC, use "http://host.docker.internal:6333"
# QDRANT_URL = "http://host.docker.internal:6333"
# For Docker on Linux use
# QDRANT_URL = "http://172.17.0.1:6333"
QDRANT_URL = "http://host.docker.internal:6333"

client = QdrantClient(QDRANT_URL)
text_model= SentenceTransformer("clip-ViT-B-32-multilingual-v1")
image_model = SentenceTransformer('clip-ViT-B-32')


@app.get("/")
def read_root():
    return FileResponse(os.path.join(BASE_DIR, "frontend/index.html"))


@app.post("/search_by_text")
async def search_by_text(query_text: str = Query(...), top_k: int = 20):
    embedding = text_model.encode(query_text)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=models.NamedVector(name="text", vector=embedding),
        limit=top_k
    )
    return results



@app.post("/search_by_text_image")
async def search_by_text_image(query_text: str = Query(...), top_k: int = 20):
    embedding = text_model.encode(query_text)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=models.NamedVector(name="image", vector=embedding),
        limit=top_k
    )
    return results



@app.post("/search_by_image")
async def search_by_image(image: UploadFile = File(...), top_k: int = 20):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    embedding = image_model.encode(img)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=models.NamedVector(name="image", vector=embedding),
        limit=top_k
    )
    return results




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("search_products:app", host="0.0.0.0", port=8002, reload=True)

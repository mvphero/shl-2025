from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import glob
import numpy as np

import os
if os.path.exists("models_cache"):
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "models_cache"
else:
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./../models_cache"

print("Loading images")
# Загрузка изображений из локальной папки
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, "images")
# Загрузка изображений
image_paths = sorted(glob.glob(f"{images_dir}/*.jpg"))

images = [Image.open(path).convert("RGB") for path in image_paths]
print("Loaded images:", len(images))
print("Loading model")
# Генерация эмбеддингов
model = SentenceTransformer('clip-ViT-B-32')
print("Generating embeddings")
embeddings = model.encode(images)
print("Generated embeddings:", len(embeddings))

# PCA для снижения размерности
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

# Визуализация изображений
fig, ax = plt.subplots(figsize=(10, 8))
for (x, y), img in zip(coords, images):
    img_array = np.array(img.resize((120, 120)))  # Конвертация в numpy
    imgbox = OffsetImage(img_array, zoom=0.5)
    ab = AnnotationBbox(imgbox, (x, y), frameon=True)
    ax.add_artist(ab)

# автоматическое масштабирование области графика по координатам изображений
ax.update_datalim(coords)
ax.autoscale()

ax.grid(True)
plt.savefig("image_embeddings.png", dpi=300, bbox_inches='tight')


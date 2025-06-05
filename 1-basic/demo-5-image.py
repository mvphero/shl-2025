from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import glob
import numpy as np

import os
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./../models_cache"


# Загрузка изображений из локальной папки
image_paths = sorted(glob.glob("images/*.jpg"))
images = [Image.open(path).convert("RGB") for path in image_paths]

# Генерация эмбеддингов
model = SentenceTransformer('clip-ViT-B-32')
embeddings = model.encode(images)

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

ax.set_title("Эмбеддинги изображений через CLIP (PCA)", fontsize=25, fontweight='bold')  # set title font size
ax.grid(True)
plt.savefig("image_embeddings.png", dpi=300, bbox_inches='tight')
plt.show()

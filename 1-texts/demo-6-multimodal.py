from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import glob
import numpy as np
import pandas as pd
import os

os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./../models_cache"

# Загрузка изображений
image_paths = sorted(glob.glob("images/*.jpg"))
images = [Image.open(path).convert("RGB") for path in image_paths]

# Загрузка моделей
multimodal_model = SentenceTransformer('clip-ViT-B-32')

# Генерация эмбеддингов для изображений
img_embeddings = multimodal_model.encode(images)

# Генерация эмбеддингов для текстов
texts = ["tomato", "помидор", "fish", "человек"]
text_embeddings = multimodal_model.encode(texts)

# PCA для снижения размерности
all_embeddings = np.vstack((img_embeddings, text_embeddings))
pca = PCA(n_components=2)
coords = pca.fit_transform(all_embeddings)

img_coords = coords[:len(images)]
text_coords = coords[len(images):]

# Визуализация изображений и текстов
fig, ax = plt.subplots(figsize=(12, 9))

# Изображения
for (x, y), img in zip(img_coords, images):
    img_array = np.array(img.resize((120, 120)))
    imgbox = OffsetImage(img_array, zoom=0.5)
    ab = AnnotationBbox(imgbox, (x, y), frameon=True)
    ax.add_artist(ab)

# Тексты
for (x, y), txt in zip(text_coords, texts):
    ax.text(x, y, txt, fontsize=16, fontweight='bold', color='red', ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

ax.update_datalim(coords)
ax.autoscale()
ax.set_title("Эмбеддинги изображений и текстов через CLIP (PCA)", fontsize=25, fontweight='bold')  # set title font size
ax.grid(True)
# save as png
plt.savefig("multimodal_embeddings.png", dpi=300, bbox_inches='tight')
plt.show()
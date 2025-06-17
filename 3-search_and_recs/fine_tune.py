import pandas as pd
from sentence_transformers import (
    SentenceTransformer, InputExample, losses, evaluation
)
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import os
if os.path.exists("models_cache"):
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "models_cache"
else:
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./../models_cache"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "data/feed_with_images.csv")

# Загружаем предварительно обученную модель
model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')

# Загружаем датасет
df = pd.read_csv(DATA_CSV)

# Формируем примеры
examples = [InputExample(texts=[row['name'], row['category_name']]) for _, row in df.iterrows()]

# DataLoader'ы
train_dataloader = DataLoader(examples, shuffle=True, batch_size=32)

# Функция потерь
train_loss = losses.MultipleNegativesRankingLoss(model)

# Fine-tuning модели с evaluator'ом
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=3,
    output_path=BASE_DIR + '/fine_tuned_text_model'
)

print("✅ Fine-tuning завершён. Модель сохранена в './fine_tuned_text_model'.")
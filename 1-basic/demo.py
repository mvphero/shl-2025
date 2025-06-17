from sentence_transformers import SentenceTransformer
import os

if os.path.exists("models_cache"):
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "models_cache"
else:
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./../models_cache"

# загружаем модель
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# тексты для примера
sentences = [
    "Эмбеддинги помогают моделям понимать смысл.",
    "Сегодня отличная погода.",
    "Какой эмбеддинг лучше использовать для анализа текста?",
]

# получаем эмбеддинги
embeddings = model.encode(sentences)

# выводим результаты
for sentence, embedding in zip(sentences, embeddings):
    print(f"Текст: {sentence}")
    print(f"Размер эмбеддинга: {embedding.shape}")
    print(f"Первые 5 элементов: {embedding[:5]}\n")


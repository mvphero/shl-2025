from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader

# Загружаем исходную модель
model_original = SentenceTransformer('clip-ViT-B-32-multilingual-v1')

# Набор данных для TripletLoss
train_examples = [
    InputExample(texts=['кринж', 'что-то вызывающее смущение', 'приятная ситуация']),
    InputExample(texts=['масик', 'заботливый парень', 'невнимательный человек']),
    InputExample(texts=['тюбик', 'неуверенный парень', 'самоуверенный мужчина']),
    InputExample(texts=['скуф', 'толстый взрослый мужчина', 'стройная молодая девушка']),
    InputExample(texts=['краш', 'привлекательный человек', 'неприятный тип']),
    InputExample(texts=['чилл', 'отдых', 'стресс']),
    InputExample(texts=['вайб', 'атмосфера', 'полное отсутствие настроения']),
    InputExample(texts=['рофл', 'шутка', 'серьёзное заявление']),
    InputExample(texts=['флексить', 'танцевать или веселиться', 'грустить и скучать']),
    InputExample(texts=['хайп', 'популярность', 'неизвестность']),
]

# Создание DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)

# TripletLoss
train_loss = losses.TripletLoss(model=model_original)

# Копируем модель для дообучения
model_tuned = SentenceTransformer('clip-ViT-B-32-multilingual-v1')

# Запуск дообучения
model_tuned.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    warmup_steps=3
)

# Проверка результатов до и после дообучения
sentence_pairs = [
    ('кринж фильм','стыдный фильм'),
    ('скуф','краш'),
    ('вайб','cool'),
    ('скуф','кринж'),
]

print("Сравнение до и после дообучения:\n")

for sent1, sent2 in sentence_pairs:
    emb_original_1 = model_original.encode(sent1)
    emb_original_2 = model_original.encode(sent2)
    emb_tuned_1 = model_tuned.encode(sent1)
    emb_tuned_2 = model_tuned.encode(sent2)

    sim_original = util.cos_sim(emb_original_1, emb_original_2).item()
    sim_tuned = util.cos_sim(emb_tuned_1, emb_tuned_2).item()

    print(f"Предложения: '{sent1}' и '{sent2}'")
    print(f"- Исходная модель сходство: {sim_original:.4f}")
    print(f"- После дообучения сходство: {sim_tuned:.4f}")
    print("Сближение" if sim_tuned > sim_original else "Отдаление", "\n")

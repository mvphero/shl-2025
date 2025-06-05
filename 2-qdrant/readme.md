1) Подгрузка демо-данных
Скачайте https://storage.yandexcloud.net/shl2025/2-qdrant/data.zip и распакуйте в 2-qdrant
2) Запуск qdrant docker-compose up
админка будет доступна тут - http://localhost:6333/dashboard
2) Индексация ```python3 search_products.py```
3) Запуск сервера  ```uvicorn search_products:app --reload --port 8000``` (обычная и finetune)
4) Просмотр результатов в браузере http://localhost:8000
5) Fine tune  ```python3 fine_tune.py```
6) Индексация finе-tune модели ```python3 index_fine_tune.py```
7) Просмотр результатов в браузере http://localhost:8000/finetuned

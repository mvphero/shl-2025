# Bare-metal версия
1. Установите необходимые зависимости:

```bash
pip install -r requirements.txt
```

2. Скачайте и распакуйте демо-данные, чтобы ускорить запуск и индексацию моделей. Поместите файл в корень проекта:
   [Скачайте models\_cache.zip](https://storage.yandexcloud.net/shl2025/models_cache.zip)

Для этого выполните следующие команды:

```bash
wget https://storage.yandexcloud.net/shl2025/models_cache.zip
unzip models_cache.zip
```

3. Ознакомьтесь с файлами и запустите их поочередно:

* Сначала выполните `demo.py` и затем файлы `demo-6-*.py`.

4. После этого переходите к файлам с именами, начинающимися на `fine-tune*.py`.

# Docker image
Запускайте через Docker, чтобы избежать проблем с зависимостями и окружением.

Перейдите в родительскую папку и выполните:
Загрузка образа из файла `shl2025-amd64.tar` или `shl2025-arm64.tar` (в зависимости от вашей архитектуры).
```bash
docker load -i shl2025-arm64.tar
```
Запуск 1го примера
ARM64
```bash 
docker run -v .:/app shl2025:arm64 python3 1-basic/demo.py
```
AMD64
```bash 
docker run -v .:/app shl2025:amd64 python3 1-basic/demo.py
```

Запуск 2го примера
```bash 
docker run -v .:/app shl2025:arm64 python3 1-basic/demo-1.py
```

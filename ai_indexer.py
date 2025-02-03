#   AI-indexer 1.7 https://github.com/iximy/AI-indexer
#   indexing images using AI models for search engines with 
#   semantic threshold search and web interface
import os # Подключаем библиотеки
import json
import ollama
import chromadb
from chromadb.utils import embedding_functions
from PIL import Image
from ollama import Client

# Настройки скрипта
IMAGE_FOLDER = "./images"  # Папка с нашими фото
DB_FOLDER = "./chroma_db"  # Папка для хранения базы ChromaDB

# Инициализируем ChromaDB
chroma_client = chromadb.PersistentClient(path=DB_FOLDER)
collection = chroma_client.get_or_create_collection(
    name="image_tags",
    metadata={"hnsw:space": "cosine"},
    embedding_function=embedding_functions.DefaultEmbeddingFunction()
)

# Указываем параметры одкючения к API Ollama 
client = Client(
    host='http://localhost:11434', 
    headers={'x-some-header': 'some-value'} 
)

def generate_tags(image_path):
    """Генерируем текстовое описание наших фото промпт отправляемый для генерации описания можно отредактировать под потребности"""
    response = client.chat(
        model="llama3.2-vision:11b",  # Здесь указываем нашу модель ИИ
        messages=[
            {"role": "user", "content": "Что на этом изображении? Ответ в 20слов", "images": [image_path]}
        ]
    )
    
    return response["message"]["content"].strip()

def index_images():
    """Перебираем все изображения в папке вызываем генерацию и сохраняем их текстовое описание в ChromaDB"""
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(IMAGE_FOLDER, filename)
            tags = generate_tags(image_path)
            
            collection.add(
                ids=[filename],
                documents=[tags],
                metadatas=[{"filename": filename}]
            )
            print(f"✅ {filename}: {tags}")



if __name__ == "__main__":
    # Запускаем индексацию наших фоточек
    index_images()


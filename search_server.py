#   AI-indexer 1.7 https://github.com/iximy/AI-indexer
#   indexing images using AI models for search engines with 
#   semantic threshold search and web interface

from flask import Flask, render_template, request # Подключаем библиотеки
import chromadb

# Указываем путь где находится наша база данных
DB_FOLDER = "./chroma_db"
SCORE_THRESHOLD = 0.5  # Указываем порог отсечки для семантического поиска, где 1 более строгий поиск

# Подключаемся к ChromaDB
chroma_client = chromadb.PersistentClient(path=DB_FOLDER)
collection = chroma_client.get_or_create_collection("image_tags")

# Инициализируем Flask, не забыв указать папку наших изображений

app = Flask(__name__, static_folder='images')

def search_images(query):
    """Последовательный поиск: сначала точное совпадение, затем семантический поиск по установленому ранее порогу"""
    results = collection.query(query_texts=[query], n_results=10)# Ограничим количество фото в выдаче

    if not results["ids"]:
        return []

    exact_matches = []
    semantic_matches = []

    for i, description in enumerate(results["documents"][0]):
        filename = results["metadatas"][0][i]["filename"]
        score = results["distances"][0][i]  

        if query.lower() in description.lower():  # Проверка на точное совпадение
            exact_matches.append(filename)
        elif score <= SCORE_THRESHOLD:  # Добавляем совпадения семантического поиска согласно установленого ранее порога
            semantic_matches.append((filename, score))

    # Сортируем, чтобы наиболее подходящие изображения оказались в выдаче первыми
    semantic_matches.sort(key=lambda x: x[1])

    # Возращаем точные, затем семантические совпадения
    return exact_matches + [x[0] for x in semantic_matches]

    
# Добавляем обработку серверной ччасти для точки входа /, по запросу отдаем данные
    
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"].strip()

        if query.lower() == "exit":
            return render_template("index.html", images=[])

        matches = search_images(query)
        return render_template("index.html", images=matches)

    return render_template("index.html", images=[])

if __name__ == "__main__":
    app.run(debug=True)

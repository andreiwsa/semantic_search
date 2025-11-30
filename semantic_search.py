import sys
import os
import time
import numpy as np
import faiss
import pickle
from pathlib import Path
import torch
from datetime import datetime
import webbrowser
import threading
import json
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import INDEX_DIR, INDEX_FILE, PATHS_FILE, SEARCH_HOST, SEARCH_PORT, EMBEDDING_MODEL, PREVIEW_MAX_CHARS, SCANNER_HOST, SCANNER_PORT, EMBEDDING_HOST, EMBEDDING_PORT
from document_processor import detect_format, extract_text, sanitize_filename, get_file_preview
from embeddings import load_embedding_model, generate_embeddings_with_embedding_model

# Инициализация Flask приложения
app = Flask(__name__, template_folder='./templates')
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Измените на надежный ключ

# Глобальные переменные для модели и индекса (загружаются один раз при старте)
model = None
index = None
doc_paths = None
device = None

def load_model_and_index():
    """Загружает модель BAAI/bge-m3 и индекс один раз при старте приложения"""
    global model, index, doc_paths, device
    print(f"=== ЗАГРУЗКА МОДЕЛИ {EMBEDDING_MODEL} И ИНДЕКСА ===")
    start_time = time.time()
    
    # Пути к файлам индекса
    index_path = Path(INDEX_DIR) / INDEX_FILE
    paths_path = Path(INDEX_DIR) / PATHS_FILE
    
    # Проверка существования файлов индекса
    if not index_path.exists():
        print(f"❌ Файл индекса не найден: {index_path}")
        print(f"Убедитесь, что вы правильно указали INDEX_DIR в конфигурации")
        return False
    
    if not paths_path.exists():
        print(f"❌ Файл метаданных не найден: {paths_path}")
        print(f"Убедитесь, что вы правильно указали INDEX_DIR в конфигурации")
        return False
    
    print(f"🔍 Загрузка индекса из: {INDEX_DIR}")
    try:
        # Загрузка индекса и метаданных
        index = faiss.read_index(str(index_path))
        with open(str(paths_path), 'rb') as f:
            doc_paths = pickle.load(f)
        
        print(f"✅ Загружено {len(doc_paths)} документов")
        
        # Определение устройства для вычислений
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"⚙️ Используемое устройство: {device}")
        
        # Загрузка модели BAAI/bge-m3 для генерации эмбеддингов
        print(f"🧠 Загрузка модели: {EMBEDDING_MODEL}")
        model = load_embedding_model(device)
        
        load_time = time.time() - start_time
        print(f"✅ Модель и индекс успешно загружены за {load_time:.2f} секунд")
        return True
    except Exception as e:
        print(f"❌ Ошибка при загрузке: {str(e)}")
        return False

def search_query_web(query: str, top_k: int = 5):
    """Выполняет семантический поиск по заданному запросу для веб-интерфейса с использованием BAAI/bge-m3"""
    global model, index, doc_paths, device
    if model is None or index is None or doc_paths is None:
        print("❌ Модель или индекс не загружены")
        return None, None
    
    try:
        # Генерация эмбеддинга для запроса
        start_time = time.time()
        query_embedding = generate_embeddings_with_embedding_model(model, [query])
        emb_time = time.time() - start_time
        
        # Поиск в индексе
        start_time = time.time()
        scores, indices = index.search(np.array(query_embedding), min(top_k, len(doc_paths)))
        search_time = time.time() - start_time
        
        # Подготовка результатов
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(doc_paths) and score > 0.3:  # Порог сходства 0.3
                doc_path = doc_paths[idx]
                # Получаем краткое содержание файла
                preview = get_file_preview(doc_path, PREVIEW_MAX_CHARS)
                
                # Определение формата файла для иконки
                file_ext = Path(doc_path).suffix.lower()
                if file_ext in ['.txt', '.text']:
                    icon = '📄'
                elif file_ext in ['.html', '.htm']:
                    icon = '🌐'
                elif file_ext in ['.docx', '.doc']:
                    icon = '📝'
                elif file_ext in ['.pdf']:
                    icon = '📕'
                elif file_ext in ['.epub']:
                    icon = '📖'
                elif file_ext in ['.mobi']:
                    icon = '📓'
                else:
                    icon = '📄'
                
                results.append({
                    "rank": i+1,
                    "path": doc_path,
                    "similarity": float(score),
                    "preview": preview,
                    "icon": icon,
                    "relative_path": os.path.relpath(doc_path, start=os.path.dirname(doc_path))
                })
        
        # Подготовка метаданных для отчета
        metadata = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "total_documents_indexed": len(doc_paths),
            "top_k_requested": top_k,
            "results_count": len(results),
            "model_used": EMBEDDING_MODEL,
            "device_used": device,
            "execution_time_seconds": {
                "embedding_creation": emb_time,
                "search": search_time,
                "total": emb_time + search_time
            }
        }
        
        return results, metadata
    except Exception as e:
        print(f"❌ Ошибка при поиске: {str(e)}")
        return None, None

@app.route('/', methods=['GET', 'POST'])
def home():
    """Главная страница с формой поиска"""
    global doc_paths
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        top_k = int(request.form.get('top_k', 5))
        if not query:
            return render_template('error.html', error="Пустой запрос. Пожалуйста, введите текст для поиска.")
        
        # Выполнение поиска
        results, metadata = search_query_web(query, top_k)
        if results is None:
            return render_template('error.html', error="Ошибка при выполнении поиска. Проверьте логи сервера.")
        
        # Передаем текущее время для футера
        current_time = datetime.now().strftime("%d.%m.%Y %H:%M")
        return render_template(
            'search_results.html', 
            query=query,
            results=results,
            metadata=metadata,
            show_header=True,
            current_year=datetime.now().year,
            current_time=current_time,
            index_dir=str(INDEX_DIR),
            host=SEARCH_HOST,
            port=SEARCH_PORT,
            scanner_host=SCANNER_HOST,
            scanner_port=SCANNER_PORT,
            embedding_host=EMBEDDING_HOST,
            embedding_port=EMBEDDING_PORT,
        )
    
    # GET запрос - показываем форму поиска
    return render_template(
        'search_form.html', 
        index_dir=str(INDEX_DIR),
        model_name=EMBEDDING_MODEL.split('/')[-1],
        doc_count=len(doc_paths) if doc_paths else "Не загружено",
        port=SEARCH_PORT,
        current_year=datetime.now().year,
        scanner_host=SCANNER_HOST,
        scanner_port=SCANNER_PORT,
        embedding_host=EMBEDDING_HOST,
        embedding_port=EMBEDDING_PORT
    )

@app.route('/save-results')
def save_results():
    """Сохраняет результаты поиска в HTML файл"""
    query = request.args.get('q', '').strip()
    top_k = int(request.args.get('top_k', 5))
    if not query:
        return "❌ Пустой запрос", 400
    
    # Выполнение поиска
    results, metadata = search_query_web(query, top_k)
    if results is None:
        return "❌ Ошибка при выполнении поиска", 500
    
    # Генерация HTML-отчета без заголовка
    current_time = datetime.now().strftime("%d.%m.%Y %H:%M")
    html_report = render_template(
        'search_results.html', 
        query=query,
        results=results,
        metadata=metadata,
        show_header=False,
        current_year=datetime.now().year,
        current_time=current_time,
        index_dir=str(INDEX_DIR),
        host=SEARCH_HOST,
        port=SEARCH_PORT,
        scanner_host=SCANNER_HOST,
        scanner_port=SCANNER_PORT,
        embedding_host=EMBEDDING_HOST,
        embedding_port=EMBEDDING_PORT
    )
    
    # Сохранение HTML-файла
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = sanitize_filename(query[:50])
    report_filename = f"search_results_{timestamp}_{safe_query}.html"
    report_path = Path(INDEX_DIR) / report_filename
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    # Возвращаем файл для скачивания
    return send_file(report_path, as_attachment=True, download_name=report_filename)

@app.route('/open-file', methods=['POST'])
def open_file():
    """Открывает файл или его директорию в проводнике"""
    try:
        data = request.json
        file_path = data.get('path', '')
        if not file_path or not os.path.exists(file_path):
            return jsonify({"success": False, "error": "Файл не найден"})
        
        # Открываем директорию с файлом
        directory = os.path.dirname(file_path)
        if sys.platform == 'win32':
            os.startfile(directory)
        elif sys.platform == 'darwin':
            os.system(f'open "{directory}"')
        else:
            os.system(f'xdg-open "{directory}"')
        
        return jsonify({"success": True, "message": "Директория открыта"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/open-document', methods=['POST'])
def open_document():
    """Открывает документ в системном приложении по умолчанию"""
    try:
        data = request.json
        file_path = data.get('path', '')
        if not file_path or not os.path.exists(file_path):
            return jsonify({"success": False, "error": "Файл не найден"})
        
        # Открываем файл в системном приложении по умолчанию
        if sys.platform == 'win32':
            os.startfile(file_path)
        elif sys.platform == 'darwin':
            os.system(f'open "{file_path}"')
        else:
            os.system(f'xdg-open "{file_path}"')
        
        return jsonify({"success": True, "message": "Документ открыт"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/status')
def status():
    """Проверка статуса сервиса"""
    return jsonify({
        "status": "online",
        "index_dir": str(INDEX_DIR),
        "documents_count": len(doc_paths) if doc_paths else 0,
        "model_loaded": model is not None,
        "model_name": EMBEDDING_MODEL,
        "device": device,
        "timestamp": datetime.now().isoformat()
    })

def open_browser():
    """Открывает браузер после запуска сервера"""
    time.sleep(1)  # Ждем, пока сервер запустится
    webbrowser.open(f'http://{SEARCH_HOST}:{SEARCH_PORT}')

def show_help():
    """Показывает справку по использованию скрипта"""
    print(f"""
📚 Семантический поиск с веб-интерфейсом ({EMBEDDING_MODEL})
Использование:
  python app.py
Конфигурация:
  INDEX_DIR = {INDEX_DIR}
  Модель = {EMBEDDING_MODEL}
  Веб-сервер: http://{SEARCH_HOST}:{SEARCH_PORT}
Требуемые файлы индекса:
  • {INDEX_FILE} (векторный индекс)
  • {PATHS_FILE} (пути к документам)
Поддерживаемые форматы файлов:
  {', '.join(SUPPORTED_EXTS)}
Особенности модели {EMBEDDING_MODEL}:
  • Размерность векторов: 1024 (для BAAI/bge-m3)
  • Максимальная длина последовательности: 8192 токена
  • Поддержка 100+ языков
Дополнительные возможности:
  • Поддержка предпросмотра текстовых файлов, DOCX, PDF, EPUB и MOBI
  • Максимальное количество результатов: 1000
  • Открытие документов в приложениях по умолчанию
Управление:
  • Сервер запустится автоматически и откроет браузер
  • Для остановки сервера нажмите Ctrl+C в консоли
  • Результаты поиска можно сохранить в HTML файл через интерфейс
Интеграция:
  • API статуса: http://{SEARCH_HOST}:{SEARCH_PORT}/status
  • Связь с другими компонентами системы:
      Сканер документов: http://{SCANNER_HOST}:{SCANNER_PORT}
      Генератор эмбеддингов: http://{EMBEDDING_HOST}:{EMBEDDING_PORT}
    """)

if __name__ == '__main__':
    # Проверка необходимых зависимостей для mobi и epub
    try:
        import ebooklib
        from ebooklib import epub
        print("✅ Поддержка формата EPUB доступна")
    except ImportError:
        print("⚠️ Библиотека ebooklib не установлена. Предпросмотр EPUB будет ограничен.")
    
    try:
        import mobi
        print("✅ Поддержка формата MOBI доступна")
    except ImportError:
        print("⚠️ Библиотека mobi не установлена. Предпросмотр MOBI будет ограничен.")
    
    # Показываем справку
    show_help()
    
    # Загружаем модель и индекс
    if not load_model_and_index():
        print("""
💡 Советы по устранению проблем:""")
        print(f"1. Проверьте правильность пути INDEX_DIR в конфигурации: {INDEX_DIR}")
        print(f"2. Убедитесь, что файлы индекса существуют в указанной директории")
        print(f"3. Для создания индекса используйте скрипт генератора эмбеддингов")
        print("""
❗ Для просмотра содержимого DOCX файлов установите: pip install python-docx
❗ Для просмотра содержимого PDF файлов установите: pip install PyMuPDF
❗ Для просмотра содержимого EPUB файлов установите: pip install EbookLib
❗ Для просмотра содержимого MOBI файлов установите: pip install mobi""")
        sys.exit(1)
    
    # Создаем и запускаем поток для открытия браузера
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Запускаем Flask сервер
    print(f"""
🚀 Запуск веб-сервера на http://{SEARCH_HOST}:{SEARCH_PORT}""")
    print("Для остановки сервера нажмите Ctrl+C в консоли")
    try:
        app.run(host=SEARCH_HOST, port=SEARCH_PORT, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("""
🛑 Сервер остановлен пользователем""")
    except Exception as e:
        print(f"❌ Ошибка при запуске сервера: {str(e)}")
        sys.exit(1)
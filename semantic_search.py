import os
import sys
import pickle
import threading
import time
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import webbrowser
import datetime

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
import config
from embeddings import load_embedding_model, generate_embeddings_with_embedding_model
from document_processor import get_file_preview
from utils import ensure_directories, format_time, save_timing_data

app = Flask(__name__)
search_lock = threading.Lock()

# Глобальные переменные для кэширования
faiss_index = None
doc_paths = None
model = None
device = 'cpu'
last_load_time = 0

def load_search_data(force_reload=False):
    """Загрузка индекса и путей к документам с кэшированием"""
    global faiss_index, doc_paths, model, device, last_load_time
    
    current_time = time.time()
    
    # Проверяем, нужно ли перезагружать данные (каждые 5 минут)
    if not force_reload and current_time - last_load_time < 300 and faiss_index is not None:
        return
        
    with search_lock:
        # Проверяем наличие необходимых файлов
        index_path = config.INDEX_DIR / config.INDEX_FILE
        paths_path = config.INDEX_DIR / config.PATHS_FILE
        
        if not index_path.exists():
            raise FileNotFoundError(f"Файл индекса не найден: {index_path}")
        if not paths_path.exists():
            raise FileNotFoundError(f"Файл путей к документам не найден: {paths_path}")
            
        # Загружаем FAISS индекс
        import faiss
        faiss_index = faiss.read_index(str(index_path))
        
        # Загружаем пути к документам
        with open(paths_path, 'rb') as f:
            doc_paths = pickle.load(f)
            
        # Определяем устройство
        import torch
        device = 'cuda' if config.USE_FP16 and torch.cuda.is_available() else 'cpu'
        
        # Загружаем модель
        model = load_embedding_model(device)
        
        last_load_time = current_time

def semantic_search(query, top_k=10, threshold=0.3):
    """Выполнение семантического поиска с измерением времени"""
    global faiss_index, doc_paths, model
    
    if faiss_index is None or doc_paths is None or model is None:
        load_search_data()
        
    start_time = time.time()
    
    # Генерируем эмбеддинг для запроса
    query_embedding = generate_embeddings_with_embedding_model(model, [query])[0]
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    
    # Выполняем поиск
    scores, indices = faiss_index.search(query_embedding, top_k)
    search_time = time.time() - start_time
    
    results = []
    for i in range(len(indices[0])):
        score = scores[0][i]
        idx = indices[0][i]
        # Пропускаем результаты с низким сходством
        if score < threshold:
            continue
        if idx < len(doc_paths):
            doc_path = doc_paths[idx]
            # Получаем превью файла
            preview_start = time.time()
            preview = get_file_preview(doc_path, config.PREVIEW_MAX_CHARS)
            preview_time = time.time() - preview_start
            
            results.append({
                'rank': i + 1,
                'path': doc_path,
                'preview': preview,
                'score': float(score),
                'filename': Path(doc_path).name,
                'preview_time': preview_time
            })
    
    total_time = time.time() - start_time
    save_timing_data("semantic_search", start_time, time.time(), 1)
    
    return results, search_time, total_time

@app.route('/')
def index():
    try:
        load_search_data()
        return render_template('search_form.html', 
                             model_name=config.EMBEDDING_MODEL,
                             device=device,
                             total_docs=len(doc_paths) if doc_paths else 0,
                             now=datetime.datetime.now())
    except Exception as e:
        return render_template('error.html', 
                             error_message=f"Ошибка загрузки данных: {str(e)}",
                             recommendations=[
                                 "Проверьте, что индекс семантического поиска был создан",
                                 "Убедитесь, что файлы semantic_index.faiss и doc_paths.pkl находятся в папке index_data",
                                 "Запустите генератор эмбеддингов для создания индекса"
                             ])

@app.route('/', methods=['POST'])
def search():
    query = request.form.get('query', '')
    top_k = int(request.form.get('top_k', 10))
    threshold = float(request.form.get('threshold', 0.3))
    
    if not query.strip():
        return render_template('search_form.html', error="Пожалуйста, введите поисковый запрос")
        
    try:
        start_time = time.time()
        results, search_time, total_time = semantic_search(query, top_k, threshold)
        
        # Форматируем время для отображения
        search_time_formatted = format_time(search_time)
        total_time_formatted = format_time(total_time)
        avg_preview_time = sum(r['preview_time'] for r in results) / len(results) if results else 0
        
        # Сохраняем статистику поиска
        search_stats = {
            'query': query,
            'results_count': len(results),
            'search_time': search_time,
            'total_time': total_time,
            'avg_preview_time': avg_preview_time,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return render_template('search_results.html', 
                             query=query,
                             results=results,
                             total_results=len(results),
                             search_time=search_time_formatted,
                             total_time=total_time_formatted,
                             avg_preview_time=format_time(avg_preview_time),
                             now=datetime.datetime.now()) 
    except Exception as e:
        return render_template('error.html', 
                             error_message=f"Ошибка поиска: {str(e)}",
                             recommendations=[
                                 "Проверьте подключение к интернету",
                                 "Убедитесь, что модель эмбеддингов загружена корректно",
                                 "Попробуйте перезапустить генератор эмбеддингов"
                             ])

@app.route('/save-results', methods=['POST'])
def save_results():
    query = request.form.get('query', '')
    results_json = request.form.get('results', '[]')
    search_time = request.form.get('search_time', '0')
    total_time = request.form.get('total_time', '0')
    
    import json
    results = json.loads(results_json)
    
    # Создаем HTML файл с результатами
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"search_results_{timestamp}.html"
    filepath = config.TEMP_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Результаты поиска: {query}</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .result {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; }}
                .rank {{ font-weight: bold; color: #007acc; }}
                .path {{ color: #666; font-size: 0.9em; }}
                .preview {{ margin-top: 10px; white-space: pre-wrap; }}
                .score {{ color: #009900; font-weight: bold; }}
                .timing {{ 
                    background-color: #f8f9fa; 
                    padding: 10px; 
                    margin: 15px 0; 
                    border-radius: 5px;
                    font-family: monospace;
                }}
            </style>
        </head>
        <body>
            <h1>Результаты поиска: {query}</h1>
            <p>Дата: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <div class="timing">
                <strong>Время обработки:</strong><br>
                Поиск в индексе: {search_time}<br>
                Общее время (включая получение превью): {total_time}<br>
                Количество результатов: {len(results)}
            </div>
        """)
        
        for result in results:
            f.write(f"""
            <div class="result">
                <div class="rank">Ранг: {result['rank']}</div>
                <div class="path">Путь: {result['path']}</div>
                <div class="score">Сходство: {result['score']:.4f}</div>
                <div class="preview">{result['preview']}</div>
            </div>
            """)
            
        f.write("""
        </body>
        </html>
        """)
        
    return send_file(filepath, as_attachment=True)

@app.route('/open-file', methods=['POST'])
def open_file():
    import subprocess
    import platform
    file_path = request.form.get('file_path', '')
    if not file_path:
        return jsonify({'error': 'Путь к файлу не указан'})
    try:
        path = Path(file_path)
        if platform.system() == 'Windows':
            subprocess.run(['explorer', '/select,', str(path)], check=True)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', '-R', str(path)], check=True)
        else:  # Linux
            subprocess.run(['xdg-open', str(path.parent)], check=True)
        return jsonify({'status': 'success', 'message': 'Папка открыта'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/open-document', methods=['POST'])
def open_document():
    import subprocess
    import platform
    file_path = request.form.get('file_path', '')
    if not file_path:
        return jsonify({'error': 'Путь к файлу не указан'})
    try:
        path = Path(file_path)
        if platform.system() == 'Windows':
            os.startfile(str(path))
        elif platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', str(path)], check=True)
        else:  # Linux
            subprocess.run(['xdg-open', str(path)], check=True)
        return jsonify({'status': 'success', 'message': 'Документ открыт'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/status')
def status():
    try:
        load_search_data()
        return jsonify({
            'status': 'ready',
            'device': device,
            'total_docs': len(doc_paths) if doc_paths else 0,
            'model': config.EMBEDDING_MODEL,
            'last_load_time': datetime.datetime.fromtimestamp(last_load_time).strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/reload-data', methods=['POST'])
def reload_data():
    """Принудительная перезагрузка данных поиска"""
    try:
        load_search_data(force_reload=True)
        return jsonify({
            'status': 'success',
            'message': 'Данные успешно перезагружены',
            'total_docs': len(doc_paths) if doc_paths else 0
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/timing-stats')
def timing_stats():
    """Статистика времени выполнения операций"""
    timing_file = config.CACHE_DIR / "timing_history.json"
    if timing_file.exists():
        try:
            with open(timing_file, 'r') as f:
                timing_data = json.load(f)
            return jsonify(timing_data[-20:])  # Последние 20 записей
        except Exception as e:
            return jsonify({'error': str(e)})
    return jsonify([])

if __name__ == '__main__':
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    ensure_directories()
    
    # Открываем браузер
    webbrowser.open(f'http://{config.SEARCH_HOST}:{config.SEARCH_PORT}/')
    
    # Запускаем сервер
    app.run(host=config.SEARCH_HOST, port=config.SEARCH_PORT, debug=False)
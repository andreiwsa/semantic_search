import os
import sys
import time
import pickle
import numpy as np
import torch
import tempfile
import webbrowser
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, jsonify, send_file
import threading
from tqdm import tqdm

# Импорт из общей конфигурации
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EMBEDDING_HOST, EMBEDDING_PORT, EMBEDDING_MODEL, MAX_FILE_SIZE, SUPPORTED_EXTS, CACHE_DIR, TEMP_DIR, USE_FP16, SCANNER_HOST, SCANNER_PORT
from document_processor import detect_format, extract_text, sanitize_filename
from embeddings import load_embedding_model, generate_embeddings_with_embedding_model, save_embeddings

# Глобальные переменные для отслеживания прогресса обработки кэша
cache_processing_status = {
    'status': 'idle',  # idle, processing, completed, error
    'progress': 0,
    'current_file': '',
    'total_files': 0,
    'processed': 0,
    'result_path': '',
    'error_message': '',
    'start_time': 0,
    'end_time': 0
}

# Инициализация Flask приложения
app = Flask(__name__, template_folder='./templates')
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['UPLOAD_FOLDER'] = str(TEMP_DIR)

# Загрузка модели BAAI/bge-m3
print("🧠 Загрузка модели BAAI/bge-m3 для создания эмбеддингов...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"⚙️ Используемое устройство: {device}")
embedding_model = load_embedding_model(device)

def process_cache_embeddings_worker(cache_path, output_path):
    """Фоновый процесс генерации эмбеддингов для всех документов из кэша"""
    global cache_processing_status
    try:
        # Загрузка кэша сканирования
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        if not cache_data:
            cache_processing_status.update({
                'status': 'error',
                'error_message': 'Не удалось загрузить кэш сканирования',
                'end_time': time.time()
            })
            return False
        
        texts = cache_data['texts']
        doc_paths = cache_data['doc_paths']
        
        cache_processing_status.update({
            'status': 'processing',
            'total_files': len(texts),
            'start_time': time.time(),
            'error_message': ''
        })
        
        print(f"🧠 Генерация эмбеддингов для {len(texts)} документов...")
        
        # Пакетная обработка для отображения прогресса
        batch_size = 8  # Уменьшаем размер батча для обработки длинных документов
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            if cache_processing_status['status'] != 'processing':
                print("❌ Обработка прервана")
                return False
            
            batch_texts = texts[i:i+batch_size]
            batch_paths = doc_paths[i:i+batch_size]
            
            # Обновление прогресса
            if batch_paths:
                cache_processing_status['current_file'] = Path(batch_paths[0]).name
            cache_processing_status['processed'] = i
            cache_processing_status['progress'] = int((i / max(1, len(texts))) * 100)
            
            # Генерация эмбеддингов для пакета
            batch_embeddings = generate_embeddings_with_embedding_model(embedding_model, batch_texts)
            all_embeddings.extend(batch_embeddings.tolist())
        
        # Сохранение результатов
        result_path = save_embeddings(output_path, doc_paths, np.array(all_embeddings))
        
        # Обновление статуса по завершению
        cache_processing_status.update({
            'status': 'completed',
            'end_time': time.time(),
            'result_path': str(result_path),
            'processed': len(texts),
            'progress': 100,
            'current_file': 'Готово'
        })
        
        print(f"✅ Эмбеддинги успешно сохранены в {result_path}")
        return True
    except Exception as e:
        cache_processing_status.update({
            'status': 'error',
            'error_message': str(e),
            'end_time': time.time()
        })
        print(f"❌ Ошибка при обработке кэша: {str(e)}")
        raise

@app.route('/')
def index():
    """Главная страница с выбором режима работы"""
    return render_template(
        'embedding_generator.html',
        supported_exts_single=SUPPORTED_EXTS,
        current_year=time.localtime().tm_year,
        port=EMBEDDING_PORT,
        scanner_host=SCANNER_HOST,
        scanner_port=SCANNER_PORT
    )

@app.route('/generate_embeddings', methods=['POST'])
def generate_embeddings_endpoint():
    """Эндпоинт для генерации эмбеддингов для отдельных файлов"""
    if 'files' not in request.files:
        return jsonify({'error': 'Файлы не были загружены'}), 400
    
    files = request.files.getlist('files')
    results = []
    
    # Если нет файлов
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'Не выбраны файлы для обработки'}), 400
    
    for file in files:
        filename = secure_filename(file.filename)
        ext = Path(filename).suffix.lower()
        
        # Проверка расширения
        if ext not in SUPPORTED_EXTS:
            results.append({
                'filename': filename,
                'error': f'Неподдерживаемый формат файла. Поддерживаются: {", ".join(SUPPORTED_EXTS)}'
            })
            continue
        
        # Сохранение файла во временную директорию
        file_path = TEMP_DIR / filename
        file.save(str(file_path))
        
        try:
            # Определение формата и извлечение текста
            file_format = detect_format(file_path)
            if not file_format:
                results.append({
                    'filename': filename,
                    'error': 'Не удалось определить формат файла'
                })
                continue
            
            text = extract_text(file_path)
            if not text:
                results.append({
                    'filename': filename,
                    'error': 'Не удалось извлечь текст из файла'
                })
                continue
            
            # Генерация эмбеддинга
            embeddings = generate_embeddings_with_embedding_model(embedding_model, [text])
            results.append({
                'filename': filename,
                'format': file_format.upper(),
                'embedding': embeddings[0].tolist() if isinstance(embeddings[0], np.ndarray) else embeddings[0],
                'text_length': len(text),
                'vector_dimension': len(embeddings[0]) if isinstance(embeddings[0], (list, np.ndarray)) else 0
            })
        except Exception as e:
            results.append({
                'filename': filename,
                'error': f'Ошибка обработки: {str(e)}'
            })
        finally:
            # Удаление временного файла
            if file_path.exists():
                file_path.unlink()
    
    return jsonify({'results': results})

@app.route('/start_cache_processing', methods=['POST'])
def start_cache_processing():
    """Запуск обработки кэша в фоновом режиме"""
    global cache_processing_status
    if cache_processing_status['status'] == 'processing':
        return jsonify({'success': False, 'message': 'Обработка кэша уже запущена'})
    
    if 'cache_file' not in request.files:
        return jsonify({'success': False, 'message': 'Файл кэша не был загружен'}), 400
    
    cache_file = request.files['cache_file']
    if cache_file.filename == '':
        return jsonify({'success': False, 'message': 'Не выбран файл кэша'}), 400
    
    # Создаем директорию кэша, если она не существует
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Сохранить загруженный файл кэша во временную директорию
    cache_path = CACHE_DIR / secure_filename(cache_file.filename)
    cache_file.save(str(cache_path))
    
    # Путь для сохранения результата
    output_path = CACHE_DIR / 'embeddings_cache.pkl'
    
    # Сброс статуса
    cache_processing_status.update({
        'status': 'idle',
        'progress': 0,
        'current_file': '',
        'total_files': 0,
        'processed': 0,
        'result_path': '',
        'error_message': '',
        'start_time': 0,
        'end_time': 0
    })
    
    # Запуск обработки в отдельном потоке
    threading.Thread(
        target=process_cache_embeddings_worker,
        args=(cache_path, output_path),
        daemon=True
    ).start()
    
    return jsonify({'success': True})

@app.route('/cache_status')
def get_cache_status():
    """Получение текущего статуса обработки кэша"""
    global cache_processing_status
    # Вычисление примерного оставшегося времени
    remaining_time = "-"
    if cache_processing_status['status'] == 'processing' and cache_processing_status['start_time'] > 0 and cache_processing_status['processed'] > 0:
        elapsed = time.time() - cache_processing_status['start_time']
        files_per_sec = cache_processing_status['processed'] / elapsed if elapsed > 0 else 0
        remaining_files = cache_processing_status['total_files'] - cache_processing_status['processed']
        if files_per_sec > 0:
            remaining_seconds = remaining_files / files_per_sec
            if remaining_seconds < 60:
                remaining_time = f"{int(remaining_seconds)} сек"
            else:
                remaining_time = f"{int(remaining_seconds/60)} мин"
    return jsonify({
        'status': cache_processing_status['status'],
        'progress': cache_processing_status['progress'],
        'current_file': cache_processing_status['current_file'],
        'total_files': cache_processing_status['total_files'],
        'processed': cache_processing_status['processed'],
        'remaining_time': remaining_time,
        'result_path': cache_processing_status['result_path'],
        'error_message': cache_processing_status['error_message'],
        'start_time': cache_processing_status['start_time'],
        'end_time': cache_processing_status['end_time']
    })

@app.route('/download_embeddings_cache')
def download_embeddings_cache():
    """Скачивание обработанного кэша эмбеддингов"""
    global cache_processing_status
    if cache_processing_status['status'] != 'completed' or not cache_processing_status['result_path']:
        return jsonify({'error': 'Нет готового файла для скачивания'}), 404
    
    try:
        return send_file(
            cache_processing_status['result_path'],
            as_attachment=True,
            download_name='embeddings_cache.pkl',
            mimetype='application/octet-stream'
        )
    except Exception as e:
        return jsonify({'error': f'Ошибка при скачивании файла: {str(e)}'}), 500

@app.route('/status')
def status():
    """Проверка статуса сервиса"""
    global cache_processing_status
    return jsonify({
        "status": "online",
        "model": EMBEDDING_MODEL,
        "device": device,
        "cache_processing": cache_processing_status['status'],
        "timestamp": time.time()
    })

def open_browser():
    """Открывает браузер после запуска сервера"""
    time.sleep(1)  # Ждем, пока сервер запустится
    webbrowser.open(f'http://{EMBEDDING_HOST}:{EMBEDDING_PORT}')

def show_help():
    """Показывает справку по использованию скрипта"""
    print(f"""
📚 Генератор эмбеддингов с веб-интерфейсом (BAAI/bge-m3)
Использование:
  python app.py
Конфигурация:
  Модель = {EMBEDDING_MODEL}
  Веб-сервер: http://{EMBEDDING_HOST}:{EMBEDDING_PORT}
Поддерживаемые форматы файлов:
  {', '.join(SUPPORTED_EXTS)}
Максимальный размер файла: {MAX_FILE_SIZE / (1024*1024)} МБ
Временная директория: {TEMP_DIR}
Кэш-директория: {CACHE_DIR}
Особенности модели BAAI/bge-m3:
  • Размерность векторов: 1024
  • Максимальная длина последовательности: 8192 токена
  • Поддержка 100+ языков
Управление:
  • Сервер запустится автоматически и откроет браузер
  • Для остановки сервера нажмите Ctrl+C в консоли
    """)

if __name__ == '__main__':
    # Проверка необходимых зависимостей для mobi и epub
    try:
        import ebooklib
        from ebooklib import epub
        print("✅ Поддержка формата EPUB доступна")
    except ImportError:
        print("⚠️ Библиотека ebooklib не установлена. Установите: pip install EbookLib")
    
    try:
        import mobi
        print("✅ Поддержка формата MOBI доступна")
    except ImportError:
        print("⚠️ Библиотека mobi не установлена. Установите: pip install mobi")
    
    # Показываем справку
    show_help()
    
    # Создаем и запускаем поток для открытия браузера
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Запуск Flask приложения
    print(f"""
🚀 Запуск веб-сервера генератора эмбеддингов на http://{EMBEDDING_HOST}:{EMBEDDING_PORT}""")
    print("Для остановки сервера нажмите Ctrl+C в консоли")
    try:
        app.run(host=EMBEDDING_HOST, port=EMBEDDING_PORT, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("""
🛑 Сервер остановлен пользователем""")
    except Exception as e:
        print(f"❌ Ошибка при запуске сервера: {str(e)}")
        sys.exit(1)
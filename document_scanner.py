import os
import sys
import time
import hashlib
import pickle
import json
import threading
import webbrowser
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from tqdm import tqdm

# Импорт из общей конфигурации
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SCANNER_HOST, SCANNER_PORT, INDEX_DIR, DEFAULT_ROOT_DIR, MIN_FILE_SIZE, MAX_TEXT_LEN, USE_CACHE, SUPPORTED_EXTS, SCAN_CACHE_FILE
from document_processor import detect_format, extract_text, get_file_hash, is_valid_file, clean_text

# Глобальные переменные для отслеживания прогресса сканирования
scan_status = {
    'status': 'idle',  # idle, scanning, completed, error
    'progress': 0,
    'current_file': '',
    'total_files': 0,
    'processed': 0,
    'skipped_small': 0,
    'skipped_empty': 0,
    'skipped_dupes': 0,
    'result_path': '',
    'error_message': '',
    'start_time': 0,
    'end_time': 0
}

# Инициализация Flask приложения
app = Flask(__name__, template_folder='./templates')

def load_scan_cache(index_dir):
    """Загрузка кэша сканирования"""
    scan_cache_path = Path(index_dir) / SCAN_CACHE_FILE
    if not scan_cache_path.exists():
        return None
    try:
        with open(scan_cache_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"⚠️ Ошибка при загрузке кэша: {str(e)}")
        return None

def scan_documents(root_dir, index_dir, min_file_size, max_text_len, use_cache):
    """Основная функция сканирования документов"""
    global scan_status
    try:
        root = Path(root_dir)
        index_path = Path(index_dir)
        if not root.is_dir():
            raise Exception(f"Папка не найдена: {root_dir}")
        index_path.mkdir(parents=True, exist_ok=True)
        scan_cache_path = index_path / SCAN_CACHE_FILE
        
        # Поддерживаемые расширения
        supported_exts = SUPPORTED_EXTS
        
        # Обновление статуса
        scan_status.update({
            'status': 'scanning',
            'start_time': time.time(),
            'error_message': ''
        })
        
        # Загрузка существующего кэша если есть
        scan_cache = load_scan_cache(index_dir) if use_cache else None
        
        # Сбор всех файлов с подходящими расширениями
        all_files = []
        for ext in supported_exts:
            all_files.extend(root.rglob(f'*{ext}'))
        scan_status['total_files'] = len(all_files)
        
        # Определение файлов для обработки при инкрементальном обновлении
        files_to_process = all_files
        removed_files = []
        doc_paths = []
        texts = []
        seen_hashes = set()
        
        if scan_cache and use_cache:
            # Загружаем данные из кэша
            doc_paths = scan_cache['doc_paths'][:]
            texts = scan_cache['texts'][:]
            seen_hashes = set(scan_cache['seen_hashes'])
            
            # Определяем файлы для обновления
            cached_paths = set(scan_cache['doc_paths'])
            cached_mtimes = {}
            for path in scan_cache['doc_paths']:
                p = Path(path)
                if p.exists():
                    cached_mtimes[path] = p.stat().st_mtime
            
            files_to_add = []
            files_to_recheck = []
            for file_path in all_files:
                str_path = str(file_path)
                if str_path not in cached_paths:
                    files_to_add.append(file_path)
                elif str_path in cached_mtimes:
                    current_mtime = file_path.stat().st_mtime
                    if current_mtime > cached_mtimes[str_path] + 1:
                        files_to_recheck.append(file_path)
            
            # Проверка удаленных файлов
            removed_files = [path for path in cached_paths if not Path(path).exists()]
            for removed_path in removed_files:
                if removed_path in doc_paths:
                    idx = doc_paths.index(removed_path)
                    doc_paths.pop(idx)
                    texts.pop(idx)
            
            files_to_process = files_to_add + files_to_recheck
        
        # Обработка файлов
        processed_count = 0
        skipped_small = 0
        skipped_empty = 0
        skipped_dupes = 0
        scan_status.update({
            'processed': 0,
            'skipped_small': 0,
            'skipped_empty': 0,
            'skipped_dupes': 0,
            'current_file': ''
        })
        
        for i, file_path in enumerate(files_to_process):
            if scan_status['status'] != 'scanning':
                break
            scan_status['current_file'] = str(file_path.relative_to(root))
            scan_status['progress'] = int((i + 1) / max(1, len(files_to_process)) * 100)
            
            if not file_path.is_file():
                continue
            
            file_size = file_path.stat().st_size
            # Пропуск маленьких файлов
            if file_size < min_file_size:
                skipped_small += 1
                scan_status['skipped_small'] = skipped_small
                continue
            
            # Извлечение текста
            text = extract_text(file_path)
            if not text:
                skipped_empty += 1
                scan_status['skipped_empty'] = skipped_empty
                continue
            
            # Очистка текста
            text = clean_text(text)
            
            # Проверка на дубликаты
            digest = get_file_hash(text)
            if digest in seen_hashes:
                skipped_dupes += 1
                scan_status['skipped_dupes'] = skipped_dupes
                continue
            
            # Добавление нового документа
            seen_hashes.add(digest)
            doc_paths.append(str(file_path))
            texts.append(text[:max_text_len])
            processed_count += 1
            scan_status['processed'] = processed_count
        
        # Сохранение кэша
        stats = {
            'processed': processed_count,
            'skipped_small': skipped_small,
            'skipped_empty': skipped_empty,
            'skipped_dupes': skipped_dupes,
            'total_docs': len(doc_paths),
            'removed_files': len(removed_files)
        }
        
        cache_data = {
            'doc_paths': doc_paths,
            'texts': texts,
            'seen_hashes': list(seen_hashes),
            'stats': stats,
            'timestamp': time.time()
        }
        
        with open(scan_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # Обновление статуса по завершению
        scan_status.update({
            'status': 'completed',
            'end_time': time.time(),
            'result_path': str(scan_cache_path),
            'processed': processed_count,
            'skipped_small': skipped_small,
            'skipped_empty': skipped_empty,
            'skipped_dupes': skipped_dupes,
            'progress': 100
        })
        
        print(f"✅ Кэш сканирования успешно сохранен в {scan_cache_path}")
        print(f"📊 Статистика: обработано {processed_count}, пропущено: мелких-{skipped_small}, пустых-{skipped_empty}, дубликатов-{skipped_dupes}")
        return scan_cache_path
    except Exception as e:
        scan_status.update({
            'status': 'error',
            'error_message': str(e),
            'end_time': time.time()
        })
        print(f"❌ Ошибка при сканировании: {str(e)}")
        raise

@app.route('/')
def index():
    """Главная страница с настройками сканирования"""
    return render_template(
        'scanner.html',
        default_root_dir=str(DEFAULT_ROOT_DIR),
        index_dir=str(INDEX_DIR),
        min_file_size_kb=MIN_FILE_SIZE / 1024,
        max_text_len=MAX_TEXT_LEN,
        use_cache=USE_CACHE,
        supported_exts=SUPPORTED_EXTS,
        port=SCANNER_PORT,
        current_year=time.localtime().tm_year
    )

@app.route('/start_scan', methods=['POST'])
def start_scan():
    """Запуск сканирования в отдельном потоке"""
    global scan_status
    if scan_status['status'] == 'scanning':
        return jsonify({'success': False, 'message': 'Сканирование уже запущено'})
    
    # Получение параметров из формы
    try:
        root_dir = request.form.get('root_dir', str(DEFAULT_ROOT_DIR))
        index_dir = request.form.get('index_dir', str(INDEX_DIR))
        min_file_size = int(float(request.form.get('min_file_size', MIN_FILE_SIZE/1024)) * 1024)
        max_text_len = int(request.form.get('max_text_len', MAX_TEXT_LEN))
        use_cache = 'use_cache' in request.form
        
        # Сброс статуса
        scan_status.update({
            'status': 'idle',
            'progress': 0,
            'current_file': '',
            'total_files': 0,
            'processed': 0,
            'skipped_small': 0,
            'skipped_empty': 0,
            'skipped_dupes': 0,
            'result_path': '',
            'error_message': '',
            'start_time': 0,
            'end_time': 0
        })
        
        # Запуск сканирования в отдельном потоке
        threading.Thread(
            target=scan_wrapper,
            args=(root_dir, index_dir, min_file_size, max_text_len, use_cache),
            daemon=True
        ).start()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка в настройках: {str(e)}'})

def scan_wrapper(root_dir, index_dir, min_file_size, max_text_len, use_cache):
    """Обертка для обработки исключений в потоке сканирования"""
    global scan_status
    try:
        scan_documents(root_dir, index_dir, min_file_size, max_text_len, use_cache)
    except Exception as e:
        scan_status.update({
            'status': 'error',
            'error_message': str(e),
            'end_time': time.time()
        })

@app.route('/scan_status')
def get_scan_status():
    """Получение текущего статуса сканирования"""
    global scan_status
    # Вычисление примерного оставшегося времени
    remaining_time = "-"
    if scan_status['status'] == 'scanning' and scan_status['start_time'] > 0 and scan_status['processed'] > 0:
        elapsed = time.time() - scan_status['start_time']
        files_per_sec = scan_status['processed'] / elapsed if elapsed > 0 else 0
        remaining_files = scan_status['total_files'] - scan_status['processed']
        if files_per_sec > 0:
            remaining_seconds = remaining_files / files_per_sec
            if remaining_seconds < 60:
                remaining_time = f"{int(remaining_seconds)} сек"
            else:
                remaining_time = f"{int(remaining_seconds/60)} мин"
    return jsonify({
        'status': scan_status['status'],
        'progress': scan_status['progress'],
        'current_file': scan_status['current_file'],
        'total_files': scan_status['total_files'],
        'processed': scan_status['processed'],
        'skipped_small': scan_status['skipped_small'],
        'skipped_empty': scan_status['skipped_empty'],
        'skipped_dupes': scan_status['skipped_dupes'],
        'remaining_time': remaining_time,
        'result_path': scan_status['result_path'],
        'error_message': scan_status['error_message'],
        'start_time': scan_status['start_time'],
        'end_time': scan_status['end_time']
    })

@app.route('/stop_scan', methods=['POST'])
def stop_scan():
    """Остановка сканирования"""
    global scan_status
    if scan_status['status'] == 'scanning':
        scan_status['status'] = 'stopping'
        return jsonify({'success': True, 'message': 'Сканирование останавливается...'})
    return jsonify({'success': False, 'message': 'Сканирование не запущено'})

def show_help():
    """Показывает справку по использованию скрипта"""
    print(f"""
📚 Веб-интерфейс для создания кэша сканирования
Использование:
  python app.py
Конфигурация:
  Директория сканирования по умолчанию: {DEFAULT_ROOT_DIR}
  Директория сохранения кэша: {INDEX_DIR}
  Веб-сервер: http://{SCANNER_HOST}:{SCANNER_PORT}
Поддерживаемые форматы файлов:
  {', '.join(SUPPORTED_EXTS)}
Минимальный размер файла: {MIN_FILE_SIZE / 1024} КБ
Максимальная длина текста: {MAX_TEXT_LEN} символов
Управление:
  • Сервер запустится автоматически и откроет браузер
  • Для остановки сервера нажмите Ctrl+C в консоли
    """)

def open_browser():
    """Открывает браузер после запуска сервера"""
    time.sleep(1)  # Ждем, пока сервер запустится
    webbrowser.open(f'http://{SCANNER_HOST}:{SCANNER_PORT}')

if __name__ == '__main__':
    # Проверка необходимых зависимостей для mobi и epub
    try:
        import ebooklib
        from ebooklib import epub
        print("✅ Поддержка формата EPUB доступна")
    except ImportError:
        print("⚠️ Библиотека ebooklib не установлена. Поддержка EPUB будет ограничена.")
    
    try:
        import mobi
        print("✅ Поддержка формата MOBI доступна")
    except ImportError:
        print("⚠️ Библиотека mobi не установлена. Поддержка MOBI будет ограничена.")
    
    # Показываем справку
    show_help()
    
    # Создаем и запускаем поток для открытия браузера
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Запуск Flask приложения
    print(f"""
🚀 Запуск веб-сервера сканера на http://{SCANNER_HOST}:{SCANNER_PORT}""")
    print("Для остановки сервера нажмите Ctrl+C в консоли")
    try:
        app.run(host=SCANNER_HOST, port=SCANNER_PORT, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("""
🛑 Сервер остановлен пользователем""")
    except Exception as e:
        print(f"❌ Ошибка при запуске сервера: {str(e)}")
        sys.exit(1)
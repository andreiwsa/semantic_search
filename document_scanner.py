# document_scanner.py (Полностью обновленная версия с чекпоинтингом и оптимизацией)
import os
import sys
import time
import json
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for
import webbrowser
import datetime
import queue
import pickle
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import chardet  # Для детектирования кодировки

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
import config
from document_processor import extract_text, get_file_hash, clean_text, detect_format
from utils import ProcessTracker, ensure_directories, should_pause_for_daily_limit, save_checkpoint, load_checkpoint, chunk_list, format_time, save_timing_data

app = Flask(__name__)
tracker = ProcessTracker()

# Глобальные переменные для управления процессом
scan_thread = None
stop_scan = threading.Event()

def count_files_in_directory(root_dir, supported_exts):
    """Подсчитывает количество файлов, подходящих по расширению."""
    count = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if Path(file).suffix.lower() in supported_exts:
                count += 1
        # Проверяем флаг остановки
        if stop_scan.is_set():
            return count
    return count

def get_all_file_paths(root_dir):
    """Получает все пути к файлам для обработки"""
    all_file_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if Path(file).suffix.lower() in config.SUPPORTED_EXTS:
                all_file_paths.append(str(Path(root) / file))
    return all_file_paths

def process_file_batch(file_batch, existing_cache, stats, min_file_size, max_text_len):
    """Обрабатывает пакет файлов"""
    results = []
    for file_path_str in file_batch:
        if stop_scan.is_set() or should_pause_for_daily_limit():
            return results, True  # Второй параметр - флаг прерывания
        
        file_path = Path(file_path_str)
        try:
            # Проверяем размер файла
            if not file_path.exists():
                stats['skipped_error'] += 1
                print(f"Файл не существует: {file_path_str}")
                continue
                
            file_size = file_path.stat().st_size
            if file_size < min_file_size:
                stats['skipped_small'] += 1
                continue
                
            # Извлекаем текст
            try:
                text = extract_text(file_path)
            except Exception as e:
                stats['skipped_error'] += 1
                print(f"Критическая ошибка извлечения текста из {file_path_str}: {str(e)}")
                continue
                
            if not text or not isinstance(text, str) or not text.strip():
                stats['skipped_empty'] += 1
                continue
                
            # Ограничиваем длину текста
            if len(text) > max_text_len:
                text = text[:max_text_len]
                
            # Очищаем текст
            text = clean_text(text)
            if not text.strip():
                stats['skipped_empty'] += 1
                continue
                
            # Проверяем на дубликат
            text_hash = get_file_hash(text)
            if text_hash in existing_cache.get('hashes', {}):
                stats['skipped_duplicate'] += 1
                continue
                
            # Обновляем статистику по типам файлов
            ext = file_path.suffix.lower()
            stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
            
            results.append({
                'path': file_path_str,
                'text': text,
                'hash': text_hash
            })
            stats['processed'] += 1
        except Exception as e:
            stats['skipped_error'] += 1
            print(f"Ошибка обработки файла {file_path_str}: {str(e)}")
            
    return results, False

def scan_documents(root_dir, min_file_size, max_text_len, use_cache, add_timestamp):
    """Функция для сканирования документов с чекпоинтингом"""
    global stop_scan
    start_time = time.time()
    stop_scan.clear()
    
    # Проверяем директорию
    root_path = Path(root_dir)
    if not root_path.exists() or not root_path.is_dir():
        tracker.finish_task(f"Директория не найдена или не является папкой: {root_dir}")
        return
        
    # Определяем имя файла кэша и чекпоинта
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
    cache_filename = f"scan_cache_{timestamp}.pkl" if timestamp else config.SCAN_CACHE_FILE
    cache_path = config.INDEX_DIR / cache_filename
    checkpoint_file = config.CACHE_DIR / f"scan_checkpoint_{timestamp if timestamp else 'latest'}.json"
    
    # Загружаем чекпоинт если существует
    checkpoint_data = load_checkpoint(checkpoint_file) if checkpoint_file.exists() and use_cache else None
    if checkpoint_data:
        doc_paths = checkpoint_data.get('doc_paths', [])
        doc_texts = checkpoint_data.get('doc_texts', [])
        doc_hashes = checkpoint_data.get('doc_hashes', [])
        stats = checkpoint_data.get('stats', {
            'processed': 0,
            'skipped_small': 0,
            'skipped_empty': 0,
            'skipped_duplicate': 0,
            'skipped_error': 0,
            'file_types': {}
        })
        processed_count = checkpoint_data.get('processed_count', 0)
        batch_num = checkpoint_data.get('batch_num', 0)
        all_file_paths = checkpoint_data.get('all_file_paths', [])
        print(f"Возобновление сканирования с чекпоинта. Уже обработано: {processed_count} файлов")
    else:
        # Подсчет файлов
        print("Подсчет файлов...")
        tracker.update_progress(0, "Подсчет файлов...")
        all_file_paths = get_all_file_paths(root_dir)
        total_files = len(all_file_paths)
        if total_files == 0:
            tracker.finish_task("Не найдено подходящих файлов для сканирования")
            return
            
        tracker.start_task("Сканирование документов", total_files)
        print(f"Найдено {total_files} файлов для обработки.")
        
        # Загружаем кэш если используется
        existing_cache = {}
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    existing_cache = pickle.load(f)
            except Exception as e:
                print(f"Ошибка загрузки кэша: {e}. Продолжение без кэша.")
                existing_cache = {}
                
        doc_paths = []
        doc_texts = []
        doc_hashes = []
        stats = {
            'processed': 0,
            'skipped_small': 0,
            'skipped_empty': 0,
            'skipped_duplicate': 0,
            'skipped_error': 0,
            'file_types': {}
        }
        processed_count = 0
        batch_num = 0
        
    # Разбиваем файлы на пакеты
    file_batches = list(chunk_list(all_file_paths, config.BATCH_SIZE_SCAN))
    total_batches = len(file_batches)
    total_files = len(all_file_paths)
    
    # Обработка файлов партиями
    for batch_num in range(batch_num, total_batches):
        if stop_scan.is_set() or should_pause_for_daily_limit():
            # Сохраняем чекпоинт
            checkpoint_data = {
                'doc_paths': doc_paths,
                'doc_texts': doc_texts,
                'doc_hashes': doc_hashes,
                'stats': stats,
                'processed_count': processed_count,
                'batch_num': batch_num,
                'all_file_paths': all_file_paths,
                'timestamp': datetime.datetime.now().isoformat()
            }
            save_checkpoint(checkpoint_file, checkpoint_data)
            pause_msg = "достигнут суточный лимит времени работы" if should_pause_for_daily_limit() else "остановлено пользователем"
            tracker.stop_task(f"Сканирование приостановлено ({pause_msg}). Прогресс сохранен.")
            return
            
        # Обрабатываем текущий пакет
        batch_start_time = time.time()
        batch_files = file_batches[batch_num]
        existing_cache = {'hashes': dict(zip(doc_hashes, doc_paths))} if doc_hashes else {}
        batch_results, interrupted = process_file_batch(
            batch_files, 
            existing_cache, 
            stats, 
            min_file_size, 
            max_text_len
        )
        
        # Добавляем результаты пакета
        for result in batch_results:
            doc_paths.append(result['path'])
            doc_texts.append(result['text'])
            doc_hashes.append(result['hash'])
            
        processed_count += len(batch_results)
        
        # Обновляем прогресс
        elapsed_time = time.time() - start_time
        batch_elapsed = time.time() - batch_start_time
        files_per_second = len(batch_results) / batch_elapsed if batch_elapsed > 0 else 0
        progress_msg = (
            f"Обработано: {processed_count}/{total_files} файлов "
            f"({processed_count/total_files*100:.1f}%) | "
            f"Пакет {batch_num+1}/{total_batches} завершен | "
            f"Скорость: {files_per_second:.1f} файлов/сек"
        )
        tracker.update_progress(processed_count, progress_msg, stats)
        
        # Сохраняем чекпоинт после каждого пакета
        if (batch_num + 1) % max(1, config.CHECKPOINT_INTERVAL // config.BATCH_SIZE_SCAN) == 0:
            checkpoint_data = {
                'doc_paths': doc_paths,
                'doc_texts': doc_texts,
                'doc_hashes': doc_hashes,
                'stats': stats,
                'processed_count': processed_count,
                'batch_num': batch_num + 1,
                'all_file_paths': all_file_paths,
                'timestamp': datetime.datetime.now().isoformat()
            }
            save_checkpoint(checkpoint_file, checkpoint_data)
            print(f"Чекпоинт сохранен после пакета {batch_num+1}")
            
    # Сохраняем результаты
    result_data = {
        'doc_paths': doc_paths,
        'texts': doc_texts,
        'hashes': doc_hashes,
        'stats': stats,
        'timestamp': datetime.datetime.now().isoformat()
    }
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(result_data, f)
        # Удаляем чекпоинт после успешного завершения
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        end_time = time.time()
        save_timing_data("document_scanning", start_time, end_time, processed_count)
        total_processed = stats['processed']
        total_skipped = sum(v for k, v in stats.items() if k != 'processed' and k != 'file_types')
        message = (
            f"Сканирование успешно завершено за {format_time(end_time - start_time)}. "
            f"Обработано: {total_processed}, пропущено: {total_skipped}. "
            f"Файл кэша: {cache_filename}"
        )
        tracker.finish_task(message, stats)
        print(message)
    except Exception as e:
        error_message = f"Ошибка сохранения кэша: {str(e)}"
        tracker.finish_task(error_message, stats)
        print(error_message)

@app.route('/')
def index():
    return render_template('scanner.html',
                         default_dir=str(config.DEFAULT_ROOT_DIR),
                         supported_exts=config.SUPPORTED_EXTS,
                         min_file_size=config.MIN_FILE_SIZE,
                         now=datetime.datetime.now()) 

@app.route('/start-scan', methods=['POST'])
def start_scan():
    global scan_thread, stop_scan
    # Сбрасываем флаг остановки
    stop_scan.clear()
    
    root_dir = request.form.get('root_dir', str(config.DEFAULT_ROOT_DIR))
    min_file_size = int(request.form.get('min_file_size', config.MIN_FILE_SIZE))
    max_text_len = int(request.form.get('max_text_len', config.MAX_TEXT_LEN))
    use_cache = request.form.get('use_cache', 'true') == 'true'
    add_timestamp_raw = request.form.get('add_timestamp', 'true')
    add_timestamp = add_timestamp_raw == 'true'
    
    # Запускаем сканирование в отдельном потоке
    scan_thread = threading.Thread(
        target=scan_documents,
        args=(root_dir, min_file_size, max_text_len, use_cache, add_timestamp)
    )
    scan_thread.start()
    return jsonify({'status': 'started', 'message': 'Сканирование запущено'})

@app.route('/stop-scan', methods=['POST'])
def stop_scan_route():
    global stop_scan
    stop_scan.set()
    return jsonify({'status': 'stopped', 'message': 'Сканирование остановлено'})

@app.route('/status')
def status():
    return jsonify(tracker.get_status())

@app.route('/timing-history')
def timing_history():
    """Возвращает историю времени выполнения задач"""
    timing_file = config.CACHE_DIR / "timing_history.json"
    if timing_file.exists():
        try:
            with open(timing_file, 'r') as f:
                timing_data = json.load(f)
            return jsonify(timing_data[-10:])  # Последние 10 записей
        except Exception as e:
            return jsonify({'error': str(e)})
    return jsonify([])

if __name__ == '__main__':
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    ensure_directories()
    
    # Открываем браузер
    webbrowser.open(f'http://{config.SCANNER_HOST}:{config.SCANNER_PORT}/')
    
    # Запускаем сервер
    app.run(host=config.SCANNER_HOST, port=config.SCANNER_PORT, debug=False)
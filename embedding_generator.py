# embedding_generator.py (Полностью обновленная версия с чекпоинтингом)
import os
import sys
import time
import pickle
import json
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for
import webbrowser
import numpy as np
import faiss
import tempfile
import shutil
import gzip
import datetime
from typing import List, Tuple

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
import config
from embeddings import load_embedding_model, generate_embeddings_with_embedding_model, create_faiss_index, save_index
from document_processor import extract_text, clean_text
from utils import ProcessTracker, ensure_directories, should_pause_for_daily_limit, save_checkpoint, load_checkpoint, chunk_list, format_time, save_timing_data

app = Flask(__name__)
tracker = ProcessTracker()

# Глобальные переменные для управления процессом
embedding_thread = None
stop_embedding = threading.Event()

def save_uploaded_file(file_obj):
    """Сохраняет загруженный файл во временный каталог и возвращает путь к нему."""
    if not hasattr(save_uploaded_file, 'temp_dir'):
        save_uploaded_file.temp_dir = script_dir / "temp_uploads"
        save_uploaded_file.temp_dir.mkdir(exist_ok=True)
    temp_path = save_uploaded_file.temp_dir / file_obj.filename
    file_obj.save(temp_path)
    return temp_path

def generate_embeddings_batch(model, batch_texts: List[str]) -> np.ndarray:
    """Генерация эмбеддингов для пакета текстов"""
    try:
        # Для BGE-small добавляем префикс задачи
        sentences = [f"Represent this sentence for searching relevant passages: {text}" for text in batch_texts]
        batch_embeddings = model.encode(
            sentences,
            batch_size=len(batch_texts),
            show_progress_bar=False,
            convert_to_tensor=False,
            normalize_embeddings=True
        )
        return np.array(batch_embeddings, dtype=np.float32)
    except Exception as e:
        raise e

def process_embeddings_from_cache(cache_file_path):
    """Генерация эмбеддингов из кэша сканирования с чекпоинтингом"""
    global stop_embedding
    
    start_time = time.time()
    stop_embedding.clear()
    
    # Проверяем наличие файла кэша
    cache_path = Path(cache_file_path)
    if not cache_path.exists():
        tracker.finish_task(f"Файл кэша не найден: {cache_file_path}")
        return
        
    # Определяем пути для чекпоинтов и результатов
    cache_stem = cache_path.stem
    checkpoint_file = config.CACHE_DIR / f"embedding_checkpoint_{cache_stem}.json"
    partial_index_dir = config.PARTIAL_INDEX_DIR / cache_stem
    partial_index_dir.mkdir(parents=True, exist_ok=True)
    
    # Загружаем кэш
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
    except Exception as e:
        tracker.finish_task(f"Ошибка загрузки кэша: {str(e)}")
        return
        
    # Проверяем формат кэша
    if 'texts' not in cache_data or 'doc_paths' not in cache_data:
        tracker.finish_task("Некорректный формат кэша сканирования")
        return
        
    all_texts = cache_data['texts']
    all_doc_paths = cache_data['doc_paths']
    total_texts = len(all_texts)
    
    if total_texts == 0:
        tracker.finish_task("Нет текстов для обработки")
        return
        
    # Загружаем существующий чекпоинт если есть
    checkpoint_data = load_checkpoint(checkpoint_file) if checkpoint_file.exists() else None
    
    if checkpoint_data:
        processed_indices = set(checkpoint_data.get('processed_indices', []))
        all_embeddings = checkpoint_data.get('embeddings', [])
        all_paths = checkpoint_data.get('paths', [])
        batch_num = checkpoint_data.get('batch_num', 0)
        start_index = batch_num * config.BATCH_SIZE_EMBEDDING
        
        print(f"Возобновление генерации эмбеддингов с чекпоинта. Уже обработано: {len(processed_indices)} текстов")
    else:
        processed_indices = set()
        all_embeddings = []
        all_paths = []
        batch_num = 0
        start_index = 0
        
    tracker.start_task("Генерация эмбеддингов", total_texts)
    
    # Определяем устройство
    import torch
    device = 'cuda' if config.USE_FP16 and torch.cuda.is_available() else 'cpu'
    tracker.update_progress(0, f"Загрузка модели на устройство: {device}")
    
    # Загружаем модель
    try:
        model = load_embedding_model(device)
        tracker.update_progress(0, f"Модель загружена. Начинается генерация...")
    except Exception as e:
        tracker.finish_task(f"Ошибка загрузки модели: {str(e)}")
        return
        
    # Обработка по пакетам
    try:
        for i in range(start_index, total_texts, config.BATCH_SIZE_EMBEDDING):
            if stop_embedding.is_set() or should_pause_for_daily_limit():
                # Сохраняем чекпоинт
                checkpoint_data = {
                    'processed_indices': list(processed_indices),
                    'embeddings': all_embeddings,
                    'paths': all_paths,
                    'batch_num': i // config.BATCH_SIZE_EMBEDDING,
                    'total_texts': total_texts,
                    'source_cache': str(cache_path),
                    'timestamp': datetime.datetime.now().isoformat()
                }
                save_checkpoint(checkpoint_file, checkpoint_data)
                
                pause_msg = "достигнут суточный лимит времени работы" if should_pause_for_daily_limit() else "остановлено пользователем"
                tracker.stop_task(f"Генерация эмбеддингов приостановлена ({pause_msg}). Прогресс сохранен.")
                return
                
            # Обрабатываем текущий пакет
            batch_indices = list(range(i, min(i + config.BATCH_SIZE_EMBEDDING, total_texts)))
            batch_texts = [all_texts[idx] for idx in batch_indices if idx not in processed_indices]
            batch_paths = [all_doc_paths[idx] for idx in batch_indices if idx not in processed_indices]
            
            if not batch_texts:
                continue
                
            batch_start_time = time.time()
            
            try:
                # Генерируем эмбеддинги для пакета
                batch_embeddings = generate_embeddings_batch(model, batch_texts)
                
                # Добавляем в общие списки
                all_embeddings.extend(batch_embeddings.tolist())
                all_paths.extend(batch_paths)
                processed_indices.update(batch_indices)
                
                # Обновляем прогресс
                elapsed_time = time.time() - start_time
                batch_elapsed = time.time() - batch_start_time
                texts_per_second = len(batch_texts) / batch_elapsed if batch_elapsed > 0 else 0
                
                progress_msg = (
                    f"Обработано: {len(processed_indices)}/{total_texts} текстов "
                    f"({len(processed_indices)/total_texts*100:.1f}%) | "
                    f"Пакет {batch_num+1} завершен | "
                    f"Скорость: {texts_per_second:.1f} текстов/сек"
                )
                
                tracker.update_progress(len(processed_indices), progress_msg)
                
                # Сохраняем чекпоинт через каждые N файлов
                if len(processed_indices) % config.CHECKPOINT_INTERVAL == 0:
                    checkpoint_data = {
                        'processed_indices': list(processed_indices),
                        'embeddings': all_embeddings,
                        'paths': all_paths,
                        'batch_num': i // config.BATCH_SIZE_EMBEDDING + 1,
                        'total_texts': total_texts,
                        'source_cache': str(cache_path),
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                    save_checkpoint(checkpoint_file, checkpoint_data)
                    print(f"Чекпоинт сохранен после обработки {len(processed_indices)} текстов")
                    
                batch_num += 1
                
            except Exception as e:
                error_msg = f"Ошибка генерации эмбеддингов для пакета {batch_num+1}: {str(e)}"
                print(error_msg)
                tracker.update_progress(len(processed_indices), error_msg)
                
        # Создаем и сохраняем FAISS индекс
        if not all_embeddings:
            tracker.finish_task("Не удалось сгенерировать ни одного эмбеддинга")
            return
            
        tracker.update_progress(total_texts, "Создание FAISS индекса...")
        
        # Преобразуем в numpy массив
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        
        # Создаем индекс
        index = create_faiss_index(embeddings_array)
        
        # Сохраняем индекс и пути
        index_path = config.INDEX_DIR / config.INDEX_FILE
        paths_path = config.INDEX_DIR / config.PATHS_FILE
        
        save_index(index, index_path)
        
        with open(paths_path, 'wb') as f:
            pickle.dump(all_paths, f)
            
        # Удаляем чекпоинт после успешного завершения
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            
        # Удаляем частичные индексы
        shutil.rmtree(partial_index_dir, ignore_errors=True)
            
        end_time = time.time()
        save_timing_data("embedding_generation", start_time, end_time, len(all_paths))
        
        # Формируем сообщение о завершении
        completion_msg = (
            f"Генерация эмбеддингов успешно завершена за {format_time(end_time - start_time)}. "
            f"Создан индекс для {len(all_paths)} документов из кэша: {cache_path.name}. "
            f"Средняя скорость: {len(all_paths)/(end_time-start_time):.1f} документов/сек"
        )
        
        tracker.finish_task(completion_msg)
        print(completion_msg)
        
    except Exception as e:
        error_msg = f"Критическая ошибка при генерации эмбеддингов: {str(e)}"
        print(error_msg)
        tracker.finish_task(error_msg)

@app.route('/')
def index():
    return render_template('embedding_generator.html', now=datetime.datetime.now())

@app.route('/process-cache', methods=['POST'])
def process_cache():
    global embedding_thread, stop_embedding
    
    # Проверяем, был ли загружен файл
    if 'cache_file' not in request.files:
        return jsonify({'error': 'Файл кэша не загружен'})
    file = request.files['cache_file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'})
        
    try:
        # Сохраняем загруженный файл во временный каталог
        temp_cache_path = save_uploaded_file(file)
        print(f"Загруженный файл кэша сохранен во временный каталог: {temp_cache_path}")
        
        # Сбрасываем флаг остановки
        stop_embedding.clear()
        
        # Запускаем генерацию в отдельном потоке
        embedding_thread = threading.Thread(target=process_embeddings_from_cache, args=(temp_cache_path,))
        embedding_thread.start()
        
        return jsonify({'status': 'started', 'message': f'Генерация эмбеддингов запущена из кэша: {file.filename}'})
    except Exception as e:
        error_msg = f"Ошибка обработки загруженного файла: {str(e)}"
        print(error_msg)
        tracker.finish_task(error_msg)
        return jsonify({'error': error_msg})

@app.route('/stop-generation', methods=['POST'])
def stop_generation():
    global stop_embedding
    stop_embedding.set()
    return jsonify({'status': 'stopped', 'message': 'Генерация остановлена'})

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
    import torch
    ensure_directories()
    
    # Открываем браузер
    webbrowser.open(f'http://{config.EMBEDDING_HOST}:{config.EMBEDDING_PORT}/')
    
    # Запускаем сервер
    app.run(host=config.EMBEDDING_HOST, port=config.EMBEDDING_PORT, debug=False)
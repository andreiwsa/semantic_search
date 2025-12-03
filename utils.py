# utils.py
import os
import sys
import time
import threading
import json
import datetime
import gzip
import pickle
from pathlib import Path
import numpy as np
import chardet  # Для детектирования кодировки

# Добавляем путь к директории скрипта для импорта config
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
import config

class ProcessTracker:
    """Класс для отслеживания прогресса процессов с учетом времени"""
    def __init__(self):
        self.lock = threading.Lock()
        self.current_task = None
        self.progress = 0
        self.status = "idle"
        self.message = ""
        self.start_time = None
        self.total_items = 0
        self.processed_items = 0
        self.stats = None
        self.last_checkpoint_time = None
        self.task_start_time = None
        self.elapsed_time = 0
        self.remaining_time = 0
        self.should_pause = False
        
    def start_task(self, task_name, total_items=0):
        with self.lock:
            self.current_task = task_name
            self.progress = 0
            self.status = "running"
            self.message = f"Начало задачи: {task_name}"
            self.start_time = time.time()
            self.task_start_time = time.time()
            self.total_items = total_items
            self.processed_items = 0
            self.stats = None
            self.elapsed_time = 0
            self.remaining_time = 0
            self.last_checkpoint_time = time.time()
            self.should_pause = False
            
    def update_progress(self, processed_items, message="", stats=None):
        with self.lock:
            current_time = time.time()
            self.processed_items = processed_items
            if self.total_items > 0:
                self.progress = min(100, int((processed_items / self.total_items) * 100))
            else:
                self.progress = 0
            if message:
                self.message = message
            if stats is not None:
                self.stats = stats
            # Обновляем время
            if self.task_start_time:
                self.elapsed_time = current_time - self.task_start_time
                if self.processed_items > 0 and self.total_items > 0:
                    rate = self.processed_items / self.elapsed_time if self.elapsed_time > 0 else 0
                    remaining_items = self.total_items - self.processed_items
                    self.remaining_time = remaining_items / rate if rate > 0 else 0
                    
            # Проверяем, нужно ли приостановить работу
            self.should_pause = should_pause_for_daily_limit()
                
    def finish_task(self, message="Задача завершена", stats=None):
        with self.lock:
            self.status = "completed"
            self.message = message
            self.progress = 100
            if stats is not None:
                self.stats = stats
            self.elapsed_time = time.time() - self.task_start_time if self.task_start_time else 0
            self.remaining_time = 0
            self.should_pause = False
            
    def stop_task(self, message="Задача остановлена", stats=None):
        with self.lock:
            self.status = "stopped"
            self.message = message
            if stats is not None:
                self.stats = stats
            self.elapsed_time = time.time() - self.task_start_time if self.task_start_time else 0
            self.remaining_time = 0
            self.should_pause = False
            
    def get_status(self):
        with self.lock:
            should_pause = should_pause_for_daily_limit()
            return {
                'current_task': self.current_task,
                'progress': self.progress,
                'status': self.status,
                'message': self.message,
                'processed_items': self.processed_items,
                'total_items': self.total_items,
                'elapsed_time': self.elapsed_time,
                'remaining_time': self.remaining_time,
                'elapsed_time_formatted': format_time(self.elapsed_time),
                'remaining_time_formatted': format_time(self.remaining_time),
                'stats': self.stats,
                'should_pause': should_pause
            }

def ensure_directories():
    """Создание необходимых директорий"""
    dirs_to_create = [
        config.INDEX_DIR,
        config.CACHE_DIR,
        config.TEMP_DIR,
        config.DEFAULT_ROOT_DIR,
        config.PARTIAL_INDEX_DIR
    ]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        
def get_file_size_mb(file_path):
    """Получение размера файла в мегабайтах"""
    return os.path.getsize(file_path) / (1024 * 1024)

def format_time(seconds):
    """Форматирование времени в читаемый формат"""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    if hours > 0:
        return f"{hours}ч {minutes}м {secs}с"
    elif minutes > 0:
        return f"{minutes}м {secs}с"
    elif seconds >= 1:
        return f"{secs}с {millis}мс"
    else:
        return f"{millis}мс"
        
def should_pause_for_daily_limit():
    """Проверяет, нужно ли приостановить работу из-за суточного лимита"""
    try:
        if config.DAILY_START_TIME and config.DAILY_END_TIME:
            # Режим с фиксированным временем работы
            current_time = datetime.datetime.now().time()
            start_time = datetime.datetime.strptime(config.DAILY_START_TIME, "%H:%M").time()
            end_time = datetime.datetime.strptime(config.DAILY_END_TIME, "%H:%M").time()
            if current_time < start_time or current_time > end_time:
                return True
        else:
            # Режим с максимальным количеством часов в сутки
            if not hasattr(should_pause_for_daily_limit, 'work_start_time'):
                should_pause_for_daily_limit.work_start_time = time.time()
            elapsed_hours = (time.time() - should_pause_for_daily_limit.work_start_time) / 3600
            return elapsed_hours >= config.MAX_DAILY_WORK_HOURS
    except Exception as e:
        print(f"Ошибка проверки суточного лимита: {e}")
        return False
        
    return False
    
def save_checkpoint(checkpoint_file, data):
    """Сохраняет чекпоинт с возможностью сжатия"""
    try:
        if config.USE_COMPRESSION:
            with gzip.open(checkpoint_file, 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Ошибка сохранения чекпоинта: {e}")
        return False
        
def load_checkpoint(checkpoint_file):
    """Загружает чекпоинт с обработкой сжатия"""
    try:
        if not checkpoint_file.exists():
            return None
            
        if config.USE_COMPRESSION:
            with gzip.open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        else:
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Ошибка загрузки чекпоинта: {e}")
        return None
        
def chunk_list(lst, chunk_size):
    """Разбивает список на части заданного размера"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]
        
def compress_file(file_path):
    """Сжимает файл для экономии места"""
    try:
        with open(file_path, 'rb') as f_in:
            with gzip.open(str(file_path) + '.gz', 'wb') as f_out:
                f_out.writelines(f_in)
        os.remove(file_path)
        return True
    except Exception as e:
        print(f"Ошибка сжатия файла {file_path}: {e}")
        return False
        
def get_cache_files():
    """Получение списка файлов кэша сканирования"""
    cache_files = []
    if config.INDEX_DIR.exists():
        for file in config.INDEX_DIR.glob("scan_cache_*.pkl"):
            cache_files.append(str(file))
        # Также добавляем основной файл
        main_cache = config.INDEX_DIR / config.SCAN_CACHE_FILE
        if main_cache.exists():
            cache_files.append(str(main_cache))
    return sorted(cache_files, reverse=True)  # Сортируем по убыванию (новые первыми)
    
def cleanup_old_checkpoints(max_checkpoints=3):
    """Удаляет старые чекпоинты, оставляя только последние N"""
    if not config.KEEP_ONLY_LAST_CHECKPOINT:
        return
    try:
        # Очистка чекпоинтов сканера
        scan_checkpoints = sorted(config.INDEX_DIR.glob("scan_checkpoint_*.json"), reverse=True)
        for checkpoint in scan_checkpoints[max_checkpoints:]:
            try:
                checkpoint.unlink()
            except:
                pass
                
        # Очистка чекпоинтов эмбеддингов
        embed_checkpoints = sorted(config.CACHE_DIR.glob("embedding_checkpoint_*.json"), reverse=True)
        for checkpoint in embed_checkpoints[max_checkpoints:]:
            try:
                checkpoint.unlink()
            except:
                pass
                
        # Очистка частичных индексов
        partial_indexes = sorted(config.PARTIAL_INDEX_DIR.glob("partial_index_*.faiss"), reverse=True)
        for index_file in partial_indexes[max_checkpoints:]:
            try:
                index_file.unlink()
            except:
                pass
    except Exception as e:
        print(f"Ошибка очистки старых чекпоинтов: {e}")
        
def save_timing_data(task_name, start_time, end_time, items_processed):
    """Сохраняет данные о времени выполнения задач для анализа производительности"""
    timing_file = config.CACHE_DIR / "timing_history.json"
    timing_data = []
    if timing_file.exists():
        try:
            with open(timing_file, 'r') as f:
                timing_data = json.load(f)
        except:
            pass
            
    timing_data.append({
        'task': task_name,
        'start_time': start_time,
        'end_time': end_time,
        'duration': end_time - start_time,
        'items_processed': items_processed,
        'items_per_second': items_processed / (end_time - start_time) if (end_time - start_time) > 0 else 0,
        'timestamp': datetime.datetime.now().isoformat()
    })
    
    # Сохраняем только последние 100 записей
    timing_data = timing_data[-100:]
    try:
        with open(timing_file, 'w') as f:
            json.dump(timing_data, f, indent=2)
    except Exception as e:
        print(f"Ошибка сохранения данных о времени: {e}")
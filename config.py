# config.py
from pathlib import Path
import os
import time
import chardet  # Для детектирования кодировки

# Общие пути
BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "index_data"    # Директория для хранения индексов
TEMP_DIR = BASE_DIR / "temp"           # Временная директория
CACHE_DIR = BASE_DIR / "cache"         # Директория для кэша
TEMPLATES_DIR = BASE_DIR / "templates" # Директория шаблонов
PARTIAL_INDEX_DIR = BASE_DIR / "partial_indexes"  # Директория для частичных индексов

# --- НОВОЕ ---
# Директория для кэша моделей Hugging Face
HF_CACHE_DIR = BASE_DIR / "models_cache"
os.environ['HF_HOME'] = str(HF_CACHE_DIR)

# --- Оптимизации для оборудования (RTX 2070, 32 ГБ ОЗУ) ---
USE_FP16 = True                        # Использовать FP16 для экономии памяти GPU
MAX_WORKERS = 4                        # Оптимальное количество потоков для CPU
BATCH_SIZE_EMBEDDING = 32              # Оптимальный размер батча для RTX 2070 (8 ГБ VRAM)
EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'  # Меньше и быстрее модель для обработки 500K файлов
USE_COMPRESSION = True                 # Использовать сжатие для экономии места
KEEP_ONLY_LAST_CHECKPOINT = True       # Хранить только последний чекпоинт

# Параметры для работы с большим количеством файлов
BATCH_SIZE_SCAN = 5000                 # Размер партии для сканера
CHECKPOINT_INTERVAL = 1000             # Сохранять чекпоинт каждые 1000 файлов
MAX_DAILY_WORK_HOURS = 15.5            # Максимальное время работы в сутки (оставляем 30 мин на завершение)
DAILY_START_TIME = None                # Время начала работы (None = немедленно)
DAILY_END_TIME = None                  # Время окончания работы (None = после MAX_DAILY_WORK_HOURS)

# Параметры веб-серверов
SEARCH_HOST = '127.0.0.1'              # Хост для поиска
SEARCH_PORT = 5000                     # Порт для поиска
EMBEDDING_HOST = '127.0.0.1'           # Хост для генератора эмбеддингов
EMBEDDING_PORT = 5001                  # Порт для генератора эмбеддингов
SCANNER_HOST = '127.0.0.1'             # Хост для сканера
SCANNER_PORT = 5002                    # Порт для сканера

# Параметры обработки файлов
SUPPORTED_EXTS = ['.txt', '.text', '.html', '.htm', '.docx', '.doc', '.pdf', '.epub', '.mobi']
MAX_FILE_SIZE = 100 * 1024 * 1024 * 1024  # 100 ГБ (максимальный размер файла)
PREVIEW_MAX_CHARS = 500                # Максимальная длина превью для результатов

# Параметры сканера документов
DEFAULT_ROOT_DIR = BASE_DIR / "documents_to_index"  # Директория по умолчанию для сканирования
MIN_FILE_SIZE = 6 * 1024               # 6 КБ (минимальный размер файла для обработки)
MAX_TEXT_LEN = 2000000                 # 2 миллиона символов (максимальная длина текста)
USE_CACHE = True                       # Использовать кэш для инкрементального обновления

# Имена файлов
INDEX_FILE = 'semantic_index.faiss'   # Имя файла FAISS индекса
PATHS_FILE = 'doc_paths.pkl'           # Имя файла с путями к документам
SCAN_CACHE_FILE = 'scan_cache.pkl'     # Имя файла кэша сканирования (базовое имя)
EMBEDDINGS_CACHE_FILE = 'embeddings_cache.pkl'  # Имя файла кэша эмбеддингов

# Параметры для работы с моделью
MODEL_DOWNLOAD_TIMEOUT = 60            # Таймаут загрузки модели (секунды)
MODEL_RETRY_ATTEMPTS = 3               # Количество попыток загрузки модели
OFFLINE_MODE = False                   # Режим работы без интернета

# Создание необходимых директорий при импорте
for dir_path in [INDEX_DIR, CACHE_DIR, TEMP_DIR, PARTIAL_INDEX_DIR, HF_CACHE_DIR, DEFAULT_ROOT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
import os
from pathlib import Path

# Общие пути
BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "index_data"
TEMP_DIR = BASE_DIR / "temp"
CACHE_DIR = BASE_DIR / "cache"

# Убедимся, что директории существуют
INDEX_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Параметры модели (используем только одну модель BAAI/bge-m3 для всего проекта)
EMBEDDING_MODEL = 'BAAI/bge-m3'
USE_FP16 = True  # Использовать FP16 для ускорения (требует CUDA)

# Параметры веб-серверов
SEARCH_HOST = '127.0.0.1'
SEARCH_PORT = 5000
EMBEDDING_HOST = '127.0.0.1'
EMBEDDING_PORT = 5001
SCANNER_HOST = '127.0.0.1'
SCANNER_PORT = 5002

# Параметры обработки файлов
SUPPORTED_EXTS = ['.txt', '.text', '.html', '.htm', '.docx', '.doc', '.pdf', '.epub', '.mobi']
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 МБ
PREVIEW_MAX_CHARS = 500

# Параметры сканера документов
DEFAULT_ROOT_DIR = BASE_DIR / "documents_to_index"
MIN_FILE_SIZE = 6 * 1024  # 6 КБ
MAX_TEXT_LEN = 2000000  # 2 миллиона символов
USE_CACHE = True  # Использовать кэш для инкрементального обновления

# Имена файлов
INDEX_FILE = 'semantic_index.faiss'
PATHS_FILE = 'doc_paths.pkl'
SCAN_CACHE_FILE = 'scan_cache.pkl'
EMBEDDINGS_CACHE_FILE = 'embeddings_cache.pkl'
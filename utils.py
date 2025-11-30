import os
import re
from pathlib import Path
import html

def sanitize_filename(filename):
    """Очищает имя файла от недопустимых символов"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename[:100]  # Ограничиваем длину имени файла

def get_file_preview(file_path, max_chars=500):
    """Возвращает краткое содержание файла для предварительного просмотра"""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return "Файл не найден"
        
        # Попробуем разные кодировки для текстовых файлов
        encodings = ['utf-8', 'cp1251', 'koi8-r', 'latin-1']
        
        # Обработка текстовых файлов
        if file_path.suffix.lower() in ['.txt', '.text', '.md', '.log', '.csv', '.json', '.xml', '.html', '.htm']:
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read(max_chars * 2)  # Читаем чуть больше, чтобы потом очистить
                        # Удаляем специальные символы и лишние пробелы
                        content = re.sub(r'\s+', ' ', content).strip()
                        # Обрезаем до нужного количества символов
                        preview = content[:max_chars]
                        return preview + "..." if len(content) > max_chars else preview
                except UnicodeDecodeError:
                    continue
            return "Не удалось прочитать содержимое файла с поддерживаемыми кодировками"
        
        # Для файлов DOCX
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            try:
                import docx
                doc = docx.Document(str(file_path))
                full_text = []
                for para in doc.paragraphs:
                    full_text.append(para.text)
                content = ' '.join(full_text)
                content = re.sub(r'\s+', ' ', content).strip()
                preview = content[:max_chars]
                return preview + "..." if len(content) > max_chars else preview
            except ImportError:
                return "Установите библиотеку python-docx для просмотра содержимого DOCX файлов"
            except Exception as e:
                return f"Ошибка при чтении DOCX файла: {str(e)}"
        
        # Для PDF файлов
        elif file_path.suffix.lower() == '.pdf':
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(str(file_path))
                text = ""
                for page in doc:
                    text += page.get_text()
                    if len(text) > max_chars * 2:
                        break
                content = re.sub(r'\s+', ' ', text).strip()
                preview = content[:max_chars]
                return preview + "..." if len(content) > max_chars else preview
            except ImportError:
                return "Установите библиотеку PyMuPDF для просмотра содержимого PDF файлов"
            except Exception as e:
                return f"Ошибка при чтении PDF файла: {str(e)}"
        
        # Для остальных типов файлов
        return f"Предпросмотр для файлов типа {file_path.suffix} не поддерживается"
    except Exception as e:
        return f"Ошибка при получении предварительного просмотра: {str(e)}"

def detect_format(file_path: Path):
    """Определение формата файла на основе расширения и содержимого"""
    try:
        ext = file_path.suffix.lower()
        if ext in ['.txt', '.text']:
            return 'txt'
        elif ext in ['.html', '.htm']:
            return 'html'
        elif ext == '.docx':
            return 'docx'
        elif ext == '.doc':
            return 'doc'
        elif ext == '.pdf':
            return 'pdf'
        
        # Анализ содержимого для файлов без расширения или с неизвестным расширением
        raw = file_path.read_bytes(500)
        if len(raw) == 0:
            return None
        
        # DOCX: ZIP-архив → начинается с PK
        if raw.startswith(b'PK\x03\x04') and b'word/' in raw[:500]:
            return 'docx'
        
        # DOC: старый формат
        if raw.startswith(b'\xD0\xCF\x11\xE0'):
            return 'doc'
        
        # Попытка декодировать как текст
        try:
            text_sample = raw.decode('utf-8', errors='strict')
        except UnicodeDecodeError:
            return None
        
        # Проверка на HTML
        lower_sample = text_sample.strip().lower()
        if lower_sample.startswith(('<html', '<!doctype', '<head', '<meta', '<title')):
            return 'html'
            
        return 'txt'
    except Exception:
        return None

def extract_text(file_path: Path):
    """Извлечение текста из файла на основе его формата"""
    fmt = detect_format(file_path)
    if not fmt:
        return ''
    
    try:
        if fmt == 'txt':
            return file_path.read_text(encoding='utf-8', errors='ignore').strip()
        elif fmt == 'html':
            from bs4 import BeautifulSoup
            raw = file_path.read_bytes()
            soup = BeautifulSoup(raw, 'html.parser')
            return soup.get_text(separator=' ', strip=True)
        elif fmt == 'docx':
            import docx2txt
            return docx2txt.process(str(file_path)).strip()
        elif fmt == 'doc':
            # Упрощенная обработка DOC
            raw = file_path.read_bytes()
            cleaned = bytes(c for c in raw if 32 <= c <= 126 or c in (9, 10, 13))
            return cleaned.decode('utf-8', errors='ignore').strip()
        elif fmt == 'pdf':
            import fitz  # PyMuPDF
            doc = fitz.open(str(file_path))
            text = ""
            for page in doc:
                text += page.get_text()
            return text.strip()
    except Exception as e:
        print(f"⚠️ Ошибка при обработке {file_path}: {str(e)}")
        return ''
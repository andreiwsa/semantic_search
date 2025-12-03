import os
import sys
import hashlib
import re
import html
import mimetypes
from pathlib import Path
from bs4 import BeautifulSoup
import docx2txt
import fitz  # PyMuPDF
from ebooklib import epub
import zipfile
import chardet  # Для детектирования кодировки

def detect_format(file_path):
    """Определение формата файла на основе расширения и содержимого"""
    path = Path(file_path)
    # Проверяем по расширению
    ext = path.suffix.lower()
    if ext in ['.txt', '.text']:
        return 'text'
    elif ext in ['.html', '.htm']:
        return 'html'
    elif ext in ['.docx']:
        return 'docx'
    elif ext in ['.doc']:
        return 'doc'
    elif ext in ['.pdf']:
        return 'pdf'
    elif ext in ['.epub']:
        return 'epub'
    elif ext in ['.mobi']:
        return 'mobi'
    else:
        # Если расширение неизвестно, определяем по MIME типу
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type:
            if mime_type.startswith('text/'):
                return 'text'
            elif mime_type.startswith('application/pdf'):
                return 'pdf'
            elif mime_type == 'application/epub+zip':
                return 'epub'
        # Проверяем содержимое файла
        try:
            with open(path, 'rb') as f:
                sample = f.read(1024)
            sample_str = sample.decode('utf-8', errors='ignore')
            if '<html' in sample_str.lower() or '<head' in sample_str.lower():
                return 'html'
            elif sample_str.startswith('%PDF'):
                return 'pdf'
            elif b'PK\x03\x04' in sample and b'mimetype' in sample:
                return 'epub'
        except:
            pass
        return 'unknown'

def detect_html_encoding(content_bytes):
    """Попытка определить кодировку HTML из мета-тегов"""
    try:
        # Конвертируем в строку для поиска мета-тегов
        sample = content_bytes[:4096].decode('utf-8', errors='ignore').lower()
        
        # Поиск meta charset
        import re
        charset_match = re.search(r'<meta[^>]+charset=["\']?([^"\'\s>]+)', sample)
        if charset_match:
            return charset_match.group(1)
        
        # Поиск http-equiv
        content_type_match = re.search(r'<meta[^>]+http-equiv=["\']?content-type["\']?[^>]+content=["\'][^"\']*charset=([^"\';]+)', sample)
        if content_type_match:
            return content_type_match.group(1)
    except:
        pass
    return None

def extract_text(file_path):
    """Извлечение текста из файла в зависимости от формата"""
    path = Path(file_path)
    fmt = detect_format(path)
    
    if fmt == 'text':
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(path, 'r', encoding='cp1251') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(path, 'r', encoding='latin-1') as f:
                    content = f.read()
        return content
        
    elif fmt == 'html':
        # Сначала пробуем прочитать как бинарный файл для детектирования кодировки
        with open(path, 'rb') as f:
            content_bytes = f.read()
        
        # Пытаемся определить кодировку из мета-тегов
        detected_encoding = detect_html_encoding(content_bytes)
        
        # Если не удалось определить из мета-тегов, используем chardet
        if not detected_encoding:
            try:
                detection = chardet.detect(content_bytes)
                detected_encoding = detection['encoding']
                confidence = detection['confidence']
                if confidence < 0.7:  # Низкая уверенность
                    detected_encoding = None
            except:
                detected_encoding = None
        
        # Попробуем различные кодировки
        encodings_to_try = ['utf-8', 'cp1251', 'windows-1251', 'latin-1', 'cp1252']
        
        # Если определили кодировку, ставим ее первой
        if detected_encoding:
            encodings_to_try.insert(0, detected_encoding.lower())
        
        content = None
        for encoding in encodings_to_try:
            try:
                # Некоторые кодировки могут иметь альтернативные названия
                alt_encodings = {
                    'windows-1251': 'cp1251',
                    'windows-1252': 'cp1252'
                }
                encoding_to_use = alt_encodings.get(encoding.lower(), encoding)
                
                content = content_bytes.decode(encoding_to_use)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        
        # Если ни одна кодировка не сработала, используем 'latin-1' как последний шанс
        if content is None:
            try:
                content = content_bytes.decode('latin-1')
            except:
                # Если все попытки провалились, возвращаем пустую строку
                print(f"Не удалось декодировать HTML файл: {file_path}")
                return ""
        
        # Извлекаем текст из HTML
        try:
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text()
            return text
        except Exception as e:
            print(f"Ошибка при парсинге HTML {file_path}: {str(e)}")
            return content  # Возвращаем исходный текст если парсинг не удался
    
    elif fmt == 'docx':
        return docx2txt.process(str(path))
    
    elif fmt == 'doc':
        # Для .doc используем antiword если доступен
        import subprocess
        try:
            result = subprocess.run(['antiword', str(path)], 
                                  capture_output=True, text=True, check=True)
            return result.stdout
        except Exception as e:
            print(f"Ошибка обработки DOC файла {file_path}: {str(e)}")
            return ""
    
    elif fmt == 'pdf':
        try:
            doc = fitz.open(str(path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"Ошибка обработки PDF файла {file_path}: {str(e)}")
            return ""
    
    elif fmt == 'epub':
        try:
            book = epub.read_epub(str(path))
            text = ""
            for item in book.get_items():
                if item.get_type() == epub.ITEM_DOCUMENT:
                    content = item.get_content().decode('utf-8', errors='ignore')
                    soup = BeautifulSoup(content, 'html.parser')
                    text += soup.get_text() + "\n"
            return text
        except Exception as e:
            print(f"Ошибка обработки EPUB файла {file_path}: {str(e)}")
            return ""
    
    elif fmt == 'mobi':
        try:
            import mobi
            tempdir, filepath = mobi.extract(str(path))
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            # Удаляем временные файлы
            import shutil
            shutil.rmtree(tempdir)
            return content
        except Exception as e:
            print(f"Ошибка обработки MOBI файла {file_path}: {str(e)}")
            return ""
    
    else:
        return ""

def get_file_hash(text):
    """Генерация хеша для текста (для определения дубликатов)"""
    return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()

def clean_text(text):
    """Очистка текста от лишних символов"""
    if not isinstance(text, str):
        return ""
    
    # Удаление лишних пробелов и переносов строк
    text = re.sub(r'\s+', ' ', text)
    # Удаление HTML тегов
    text = re.sub(r'<[^>]+>', '', text)
    # Декодирование HTML entities
    text = html.unescape(text)
    return text.strip()

def sanitize_filename(filename):
    """Очистка имени файла от недопустимых символов"""
    # Заменяем недопустимые символы
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def get_file_preview(file_path, max_chars=500):
    """Получение превью содержимого файла"""
    try:
        text = extract_text(file_path)
        if not isinstance(text, str):
            return "Не удалось получить превью"
        
        text = clean_text(text)
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text
    except Exception as e:
        print(f"Ошибка получения превью для {file_path}: {str(e)}")
        return "Не удалось получить превью"
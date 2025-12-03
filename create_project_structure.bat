@echo off
setlocal

:: Получаем директорию, откуда запущен скрипт
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Создание структуры проекта в: %CD%

:: Создаём основные файлы
type nul > config.py
type nul > document_processor.py
type nul > document_scanner.py
type nul > embeddings.py
type nul > embedding_generator.py
type nul > semantic_search.py

:: Создаём папки
mkdir templates index_data cache temp 2>nul

:: Создаём HTML-шаблоны внутри templates
type nul > templates\base.html
type nul > templates\scanner.html
type nul > templates\embedding_generator.html
type nul > templates\search_form.html
type nul > templates\search_results.html
type nul > templates\error.html

echo.
echo Структура проекта успешно создана!
echo.
pause
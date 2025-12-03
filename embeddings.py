import torch
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path
import config

def load_embedding_model(device='cpu'):
    """Загрузка модели BAAI/bge-m3 с поддержкой offline-режима"""
    try:
        # Загружаем модель с параметрами для оптимизации
        model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            device=device,
            trust_remote_code=True
        )
        
        # Устанавливаем режим eval для ускорения
        model.eval()
        
        return model
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        raise

def generate_embeddings_with_embedding_model(model, texts):
    """Генерация эмбеддингов для текстов"""
    # Преобразуем в список если передан один текст
    if isinstance(texts, str):
        texts = [texts]
    
    # Подготовка текстов для модели (BGE модели ожидают специальный префикс)
    sentences = []
    for text in texts:
        # Для BGE моделей добавляем префикс задачи
        sentences.append(f"Represent this sentence for searching relevant passages: {text}")
    
    # Генерация эмбеддингов
    embeddings = model.encode(
        sentences,
        batch_size=4,
        show_progress_bar=True,
        convert_to_tensor=False,
        normalize_embeddings=True  # Нормализуем для Inner Product
    )
    
    return np.array(embeddings)

def save_embeddings(output_path, doc_paths, embeddings):
    """Сохранение эмбеддингов в файл"""
    data = {
        'doc_paths': doc_paths,
        'embeddings': embeddings
    }
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

def create_faiss_index(embeddings):
    """Создание FAISS индекса для поиска"""
    # Преобразуем в float32 если нужно
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    
    # Создаем индекс Inner Product (для косинусного сходства)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # Добавляем векторы в индекс
    index.add(embeddings)
    
    return index

def save_index(index, index_path):
    """Сохранение FAISS индекса в файл"""
    faiss.write_index(index, str(index_path))
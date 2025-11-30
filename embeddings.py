import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def load_embedding_model(device):
    """Загружает модель BAAI/bge-m3 для генерации эмбеддингов"""
    model = SentenceTransformer('BAAI/bge-m3')
    model.to(device)
    return model

def generate_embeddings_with_embedding_model(model, texts):
    """Генерирует эмбеддинги для текстов с использованием модели BAAI/bge-m3"""
    # Используем normalize_embeddings=True для Inner Product в FAISS
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def save_embeddings(output_path, doc_paths, embeddings):
    """Сохраняет эмбеддинги и пути к документам в файл"""
    import pickle
    with open(output_path, 'wb') as f:
        pickle.dump({'doc_paths': doc_paths, 'embeddings': embeddings}, f)
    return output_path

def create_faiss_index(embeddings):
    """Создает FAISS индекс для поиска по эмбеддингам"""
    dimension = embeddings.shape[1]
    # Используем IndexFlatIP для inner product (косинусное сходство для нормализованных векторов)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def save_index(index, index_path):
    """Сохраняет FAISS индекс в файл"""
    faiss.write_index(index, str(index_path))
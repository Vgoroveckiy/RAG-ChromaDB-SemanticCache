import gc
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from unstructured.partition.auto import partition

from config import Config, config

# Инициализация эмбеддингов и LLM на основе конфигурации
embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

llm = ChatOpenAI(
    model=config.LLM_MODEL,
    temperature=config.LLM_TEMPERATURE,
)

# Инициализация Text Splitter один раз (ГЛОБАЛЬНО)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
)


class VectorDatabase:
    """
    Класс для управления векторными базами данных ChromaDB.
    Используется для основного индекса документов и для кэша вопросов/ответов.
    """

    def __init__(
        self, db_path: str, cache_path: str, embeddings: HuggingFaceEmbeddings
    ) -> None:
        """
        Инициализирует пути к базам данных ChromaDB и модель эмбеддингов.
        Args:
            db_path (str): Путь для сохранения основной базы данных ChromaDB.
            cache_path (str): Путь для сохранения базы данных ChromaDB кэша.
            embeddings (HuggingFaceEmbeddings): Модель эмбеддингов для векторизации текста.
        """
        self.db_path = db_path
        self.cache_path = cache_path
        self.embeddings = embeddings
        self.db: Optional[Chroma] = None
        """Основная коллекция ChromaDB для документов."""
        self.cache_db: Optional[Chroma] = None
        """Коллекция кэша ChromaDB для кэшированных вопросов и ответов."""

    def load_or_create(self, force_recreate: bool = False) -> None:
        """
        Загружает или создает основную коллекцию ChromaDB для документов.
        Если force_recreate=True, то существующая папка коллекции будет удалена,
        а новая коллекция будет создана.
        """
        if force_recreate:
            if os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
                print(
                    f"Существующая папка основной коллекции ChromaDB удалена: {self.db_path}."
                )
            else:
                print(
                    f"Папка основной коллекции ChromaDB не найдена ({self.db_path}). Создание новой."
                )

        self.db = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings,
            collection_name="documents",
        )

        if self.db and hasattr(self.db, "_collection"):
            if self.db._collection.count() > 0:
                print(
                    f"Основная коллекция ChromaDB 'documents' загружена из {self.db_path}."
                )
            else:
                print(f"Новая основная коллекция ChromaDB создана в {self.db_path}.")

    def load_or_create_cache(self, force_recreate: bool = False) -> None:
        """
        Загружает или создает коллекцию кэша ChromaDB.
        """
        if force_recreate:
            if os.path.exists(self.cache_path):
                shutil.rmtree(self.cache_path)
                print(f"Существующая папка кэша ChromaDB удалена: {self.cache_path}.")
            else:
                print(
                    f"Папка кэша ChromaDB не найдена ({self.cache_path}). Создание новой."
                )

        self.cache_db = Chroma(
            persist_directory=self.cache_path,
            embedding_function=self.embeddings,
            collection_name="current",
        )
        if self.cache_db and hasattr(self.cache_db, "_collection"):
            if self.cache_db._collection.count() > 0:
                print(f"Переменная collection_name изменена на 'current'.")
            else:
                print(f"Переменная collection_name создана в {self.cache_path}.")

    def add_to_cache(
        self, question: str, answer: str, sources: Optional[List[str]] = None
    ) -> bool:
        """
        Добавляет вопрос-ответную пару в семантический кэш.

        Аргументы:
            question: текст вопроса, который будет кэшироваться
            answer: текст ответа, который будет кэшироваться
            sources: необязательный список имен исходных документов

        Возвращает:
            bool: True, если запись успешно добавлена, False в случае ошибки
        """
        try:
            if not self.cache_db:
                self.load_or_create_cache()
                if not self.cache_db or not hasattr(self.cache_db, "_collection"):
                    print("Ошибка: не удалось инициализировать кэш")
                    return False

            # Очистка кэша перед добавлением новой записи
            if self.cache_db and hasattr(self.cache_db, "_collection"):
                self.cleanup_expired_cache_entries(config.CACHE_TTL_DAYS)

            doc_id = str(uuid.uuid4())
            metadata = {
                "answer": answer,
                "timestamp": datetime.now().isoformat(),
                "sources": json.dumps(sources if sources is not None else []),
                "doc_id": doc_id,
            }

            if self.cache_db:  # Дополнительная проверка
                self.cache_db.add_texts(
                    texts=[question],
                    metadatas=[metadata],
                    ids=[doc_id],
                )
                print(f"Добавлена запись в кэш с ID: {doc_id}")
                return True
            return False
        except Exception as e:
            print(f"Ошибка при добавлении записи в кэш ChromaDB: {e}")
            return False

    def get_cached_answer(
        self,
        question: str,
        similarity_threshold: float = 0.1,
    ) -> Optional[str]:
        """
        Ищет кэшированный ответ на похожий вопрос.

        Аргументы:
            question: Вопрос, который нужно найти
            similarity_threshold: Максимальная дистанция L2 для совпадения (0-1)

        Возвращает:
            Optional[str]: Кэшированный ответ, если найден, иначе None

        Исключения:
            ValueError: Если вопрос пустой или недопустимый
            TypeError: Если similarity_threshold не является числом
        """
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")
        if not isinstance(similarity_threshold, (int, float)):
            raise TypeError("similarity_threshold must be numeric")

        if not self.cache_db:
            return None

        try:
            results = self.cache_db.similarity_search_with_score(query=question, k=1)

            if not results:
                return None

            doc, score = results[0]
            if score <= similarity_threshold:
                answer = (
                    doc.metadata.get("answer") if hasattr(doc, "metadata") else None
                )
                return str(answer) if answer is not None else None
            else:
                return None

        except Exception as e:
            print(f"Cache search error: {e}")
            return None

    def delete_documents(self, doc_ids: List[str]) -> None:
        """
        Удаляет документы из ChromaDB по их ID.

        Аргументы:
            doc_ids: Список ID документов для удаления

        Исключения:
            TypeError: Если doc_ids не является списком или содержит нестроковые элементы
            ValueError: Если doc_ids пустой
            RuntimeError: Если операции ChromaDB неудачны
        """
        if not isinstance(doc_ids, list):
            raise TypeError("doc_ids must be a list")
        if not all(isinstance(doc_id, str) for doc_id in doc_ids):
            raise TypeError("All doc_ids must be strings")
        if not doc_ids:
            print("Нет ID для удаления из ChromaDB.")
            return

        if not self.db:
            print("ChromaDB не инициализирована. Невозможно удалить документы.")
            return

        if not hasattr(self.db, "delete"):
            print("ChromaDB не поддерживает удаление документов.")
            return

        try:
            # Additional checks before deletion
            if not hasattr(self.db, "_collection"):
                print("Коллекция ChromaDB не доступна для удаления.")
                return

            collection = self.db._collection
            if not hasattr(collection, "count"):
                print("Коллекция не поддерживает проверку количества документов.")
                return

            # Get current count before deletion for verification
            initial_count = collection.count() if hasattr(collection, "count") else 0

            # Perform deletion
            self.db.delete(ids=doc_ids)
            print(f"Удалено {len(doc_ids)} документов из ChromaDB.")

            # Verify deletion if possible
            if hasattr(collection, "count"):
                new_count = collection.count()
                expected_count = max(0, initial_count - len(doc_ids))
                if new_count != expected_count:
                    print(
                        f"Предупреждение: Ожидалось {expected_count} документов после удаления, "
                        f"но найдено {new_count}"
                    )

        except Exception as e:
            raise RuntimeError(
                f"Ошибка при удалении документов из ChromaDB: {e}"
            ) from e

    def delete_cached_entries_by_source(self, source_file_name: str) -> None:
        """
        Удаляет записи кэша, связанные с указанным файлом-источником.

        Аргументы:
            source_file_name: Имя файла-источника, для которого нужно удалить записи

        Генерирует:
            TypeError: Если source_file_name не является строкой
            RuntimeError: Если операции с кэшем завершаются с ошибкой
        """
        if not isinstance(source_file_name, str):
            raise TypeError("source_file_name must be a string")

        if not self.cache_db:
            print("Кэш ChromaDB не инициализирован.")
            return

        try:
            # Ensure cache is loaded
            if not hasattr(self.cache_db, "_collection"):
                self.load_or_create_cache()
                if not hasattr(self.cache_db, "_collection"):
                    print("Не удалось загрузить коллекцию кэша.")
                    return

            collection = self.cache_db._collection
            if not hasattr(collection, "count"):
                print("Коллекция кэша не поддерживает операцию count")
                return

            # Get count safely
            count = collection.count() if hasattr(collection, "count") else 0
            if count == 0:
                print("Кэш пуст.")
                return

            if not hasattr(collection, "get"):
                print("Коллекция кэша не поддерживает операцию get")
                return

            # Get entries safely with type checking
            all_entries = collection.get(include=["metadatas"])
            if not all_entries or not isinstance(all_entries, dict):
                print("Не удалось получить записи кэша.")
                return

            ids = all_entries.get("ids", [])
            if not isinstance(ids, list):
                print("Неверный формат ID записей в кэше.")
                return

            metadatas = all_entries.get("metadatas", [])
            if not isinstance(metadatas, list):
                print("Неверный формат метаданных в кэше.")
                return

            ids_to_delete = []
            for i, doc_id in enumerate(ids):
                if i >= len(metadatas):
                    continue

                metadata = metadatas[i]
                if not isinstance(metadata, dict):
                    continue

                sources_str = metadata.get("sources")
                if not isinstance(sources_str, str):
                    continue

                try:
                    sources = json.loads(sources_str)
                    if isinstance(sources, list) and source_file_name in sources:
                        ids_to_delete.append(doc_id)
                except json.JSONDecodeError as e:
                    print(f"Ошибка десериализации 'sources' для {doc_id}: {e}")

            # Delete matching entries
            if ids_to_delete:
                if not hasattr(self.cache_db, "delete"):
                    print("Коллекция кэша не поддерживает операцию delete")
                    return

                try:
                    self.cache_db.delete(ids=ids_to_delete)
                    print(
                        f"Удалено {len(ids_to_delete)} записей, связанных с '{source_file_name}'."
                    )
                except Exception as e:
                    raise RuntimeError(f"Ошибка при удалении записей: {e}")
            else:
                print(f"Не найдено записей, связанных с '{source_file_name}'.")

        except Exception as e:
            raise RuntimeError(f"Ошибка при обработке кэша: {e}")

    def cleanup_expired_cache_entries(self, ttl_days: float) -> None:
        """
        Очищает просроченные записи из семантического кэша.

        Аргументы:
            ttl_days: Время жизни записей кэша в днях

        Генерирует:
            TypeError: Если ttl_days не является числом
            ValueError: Если ttl_days отрицательное
            RuntimeError: Если операции с кэшем завершаются с ошибкой
        """
        if not isinstance(ttl_days, (int, float)):
            raise TypeError("ttl_days must be numeric")
        if ttl_days < 0:
            raise ValueError("ttl_days cannot be negative")

        # Validate cache state
        if not self.cache_db:
            print("Кэш ChromaDB не инициализирован. Пропускаем очистку.")
            return

        try:
            # Ensure cache is loaded
            if not hasattr(self.cache_db, "_collection"):
                self.load_or_create_cache()
                if not hasattr(self.cache_db, "_collection"):
                    print("Не удалось загрузить коллекцию кэша.")
                    return

            collection = self.cache_db._collection
            if not hasattr(collection, "count"):
                print("Коллекция кэша не поддерживает операцию count")
                return

            # Get count safely
            count = collection.count() if hasattr(collection, "count") else 0
            if count == 0:
                print("Кэш пуст, нет просроченных записей для очистки.")
                print("--- Очистка кэша завершена ---")
                return

            current_time = datetime.now()
            expiration_threshold = current_time - timedelta(days=ttl_days)
            ids_to_delete = []

            if not hasattr(collection, "get"):
                print("Коллекция кэша не поддерживает операцию get")
                return

            # Get entries safely with type checking
            all_entries = collection.get(include=["metadatas"])
            if not all_entries or not isinstance(all_entries, dict):
                print("Не удалось получить записи кэша для очистки.")
                return

            ids = all_entries.get("ids", [])
            if not isinstance(ids, list):
                print("Неверный формат ID записей в кэше.")
                return

            metadatas = all_entries.get("metadatas", [])
            if not isinstance(metadatas, list):
                print("Неверный формат метаданных в кэше.")
                return

            # Process entries
            for i, doc_id in enumerate(ids):
                if i >= len(metadatas):
                    continue

                metadata = metadatas[i]
                if not isinstance(metadata, dict):
                    continue

                timestamp_str = metadata.get("timestamp")
                if not isinstance(timestamp_str, str):
                    continue

                try:
                    cache_datetime = datetime.fromisoformat(timestamp_str)
                    if cache_datetime < expiration_threshold:
                        ids_to_delete.append(doc_id)
                except ValueError as e:
                    print(
                        f"Предупреждение: Неверный формат timestamp в кэше для {doc_id}: {timestamp_str}. Ошибка: {e}"
                    )

            # Delete expired entries
            if ids_to_delete:
                if not hasattr(self.cache_db, "delete"):
                    print("Коллекция кэша не поддерживает операцию delete")
                    return

                try:
                    self.cache_db.delete(ids=ids_to_delete)
                    print(
                        f"Удалено {len(ids_to_delete)} просроченных записей из семантического кэша."
                    )
                except Exception as e:
                    raise RuntimeError(f"Ошибка при удалении записей из кэша: {e}")
            else:
                print("Не найдено просроченных записей в семантическом кэше.")

        except Exception as e:
            raise RuntimeError(f"Ошибка при обработке кэша: {e}")

        print("--- Очистка кэша завершена ---")


def get_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    if file_path.lower().endswith(".json"):
        with open(
            file_path, "r", encoding="utf-8-sig"
        ) as f:  # Используем utf-8-sig для обработки BOM
            try:
                data = json.load(f)
                canonical_content = json.dumps(
                    data, sort_keys=True, ensure_ascii=False
                ).encode("utf-8")
                hasher.update(canonical_content)
            except UnicodeDecodeError:
                print(
                    f"Ошибка кодировки при чтении {file_path}. Проверяйте UTF-8 без BOM."
                )
                raise
    else:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def get_canonical_json_content(file_path: str) -> str:
    """
    Возвращает каноническое содержимое JSON-файла для сравнения.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return json.dumps(data, sort_keys=True, ensure_ascii=False)
        except UnicodeDecodeError:
            print(f"Ошибка кодировки при чтении {file_path}. Проверяйте UTF-8 без BOM.")
            raise


def get_chroma_id_from_path_and_hash(file_path: str, file_hash: str) -> str:
    """Генерирует детерминированный ID для ChromaDB на основе пути к файлу и его хэша."""
    return hashlib.sha256(f"{file_path}:{file_hash}".encode()).hexdigest()


def update_document_in_chroma(
    vector_db: VectorDatabase,
    file_path: str,
    full_text_content: str,
    metadata: dict,
    current_file_hash: str,
    current_last_modified: float,
    stored_doc_id: Optional[str] = None,
) -> tuple[List[Document], List[str]]:
    """
    Обновляет документ в ChromaDB, удаляя старые чанки и добавляя новые.

    Аргументы:
        vector_db: Экземпляр VectorDatabase
        file_path: Путь к файлу, который обновляется
        full_text_content: Полное текстовое содержимое документа
        metadata: Метаданные для документа
        current_file_hash: Текущий хэш файла
        current_last_modified: Метка времени последнего изменения
        stored_doc_id:.Optional существующий ID документа, который обновляется

    Возвращает:
        tuple: (Список новых Document chunks, Список новых ID ChromaDB)

    Вызывает:
        TypeError: Если параметры ввода недействительны
        RuntimeError: Если операции ChromaDB завершаются неудачно
    """
    if not isinstance(vector_db, VectorDatabase):
        raise TypeError("vector_db must be a VectorDatabase instance")
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")
    if not isinstance(full_text_content, str):
        raise TypeError("full_text_content must be a string")
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict")

    base_file_name_for_cache_invalidation = os.path.basename(file_path.split("#")[0])
    new_chunks: List[Document] = []
    new_chroma_ids: List[str] = []

    # Delete old chunks if they exist
    if stored_doc_id and vector_db.db and hasattr(vector_db.db, "_collection"):
        try:
            ids_to_delete = []
            if "#" in file_path:
                if hasattr(vector_db.db, "get"):
                    existing_docs_for_path = vector_db.db.get(
                        where={"file_path": file_path},
                        include=["metadatas", "documents"],
                    )
                    if existing_docs_for_path and isinstance(
                        existing_docs_for_path, dict
                    ):
                        ids_to_delete.extend(existing_docs_for_path.get("ids", []))
            else:
                if hasattr(vector_db.db, "get"):
                    existing_docs_for_source = vector_db.db.get(
                        where={"source": base_file_name_for_cache_invalidation},
                        include=["metadatas", "documents"],
                    )
                    if existing_docs_for_source and isinstance(
                        existing_docs_for_source, dict
                    ):
                        ids_to_delete.extend(existing_docs_for_source.get("ids", []))

            if ids_to_delete:
                vector_db.delete_documents(ids_to_delete)
                print(
                    f"Удалены старые ChromaDB чанки для {os.path.basename(file_path)}."
                )
        except Exception as e:
            raise RuntimeError(
                f"Ошибка при удалении старых чанков для {file_path}: {e}"
            )

    # Create new document chunks
    new_chunks = text_splitter.create_documents(
        [full_text_content], metadatas=[metadata]
    )
    if not new_chunks:
        print(f"Нет записей для добавления для {os.path.basename(file_path)}.")
        return new_chunks, new_chroma_ids

    texts = [chunk.page_content for chunk in new_chunks]
    metadatas = [chunk.metadata for chunk in new_chunks]
    ids = [str(uuid.uuid4()) for _ in new_chunks]

    # Ensure database is initialized
    if not vector_db.db:
        vector_db.load_or_create()

    if not vector_db.db or not hasattr(vector_db.db, "add_texts"):
        print("Ошибка: не удалось инициализировать ChromaDB для добавления текстов")
        return new_chunks, new_chroma_ids

    try:
        vector_db.db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        new_chroma_ids.extend(ids)
        print(f"Добавлено {len(new_chroma_ids)} новых чанков в ChromaDB.")
    except Exception as e:
        raise RuntimeError(f"Ошибка при добавлении текстов в ChromaDB: {e}")

    return new_chunks, new_chroma_ids


def process_catalog_data(file_path: str, vector_db: VectorDatabase) -> List[Document]:
    """
    Обрабатывает JSON файл каталога в документы и индексирует их в ChromaDB.

    Аргументы:
        file_path: Путь к JSON файлу каталога
        vector_db: Экземпляр VectorDatabase для использования

    Возвращает:
        Список обработанных объектов Document

    Исключения:
        TypeError: Если входные параметры неверны
        RuntimeError: Если обработка завершилась ошибкой
    """
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")
    if not isinstance(vector_db, VectorDatabase):
        raise TypeError("vector_db must be a VectorDatabase instance")

    documents_for_chroma: List[Document] = []
    try:
        # Get file metadata
        current_file_hash = get_file_hash(file_path)
        current_last_modified = os.path.getmtime(file_path)
        current_canonical_content = get_canonical_json_content(file_path)
        base_filename = os.path.basename(file_path)

        # Check if file is unchanged
        file_unchanged = False
        existing_file_hash = None
        existing_last_modified = None
        existing_canonical_content = None

        # Validate ChromaDB state
        if not vector_db.db or not hasattr(vector_db.db, "_collection"):
            print("ChromaDB не инициализирована. Начинаем с чистого индекса.")
        else:
            collection = vector_db.db._collection
            if not hasattr(collection, "count"):
                print("Коллекция ChromaDB не поддерживает операцию count")
            else:
                count = collection.count() if hasattr(collection, "count") else 0
                if count > 0:
                    if not hasattr(collection, "get"):
                        print("Коллекция ChromaDB не поддерживает операцию get")
                    else:
                        try:
                            existing_docs = collection.get(
                                where={"source": base_filename},
                                include=["metadatas", "documents"],
                            )
                            if (
                                existing_docs
                                and isinstance(existing_docs, dict)
                                and "metadatas" in existing_docs
                                and isinstance(existing_docs["metadatas"], list)
                                and len(existing_docs["metadatas"]) > 0
                            ):
                                first_metadata = existing_docs["metadatas"][0]
                                if isinstance(first_metadata, dict):
                                    existing_file_hash = first_metadata.get("file_hash")
                                    existing_last_modified = first_metadata.get(
                                        "last_modified", 0
                                    )
                                    existing_canonical_content = first_metadata.get(
                                        "canonical_content", ""
                                    )

                                    if (
                                        existing_canonical_content
                                        == current_canonical_content
                                        and isinstance(
                                            existing_last_modified, (int, float)
                                        )
                                        and isinstance(
                                            current_last_modified, (int, float)
                                        )
                                        and abs(
                                            existing_last_modified
                                            - current_last_modified
                                        )
                                        < 60.0
                                    ):
                                        file_unchanged = True
                        except Exception as e:
                            raise RuntimeError(
                                f"Ошибка при проверке существующих документов: {e}"
                            )

        if file_unchanged:
            print(f"Документ не изменился: {base_filename}.")
            # Загружаем существующие документы
            all_json_chunks = None
            if vector_db.db and hasattr(vector_db.db, "get"):
                all_json_chunks = vector_db.db.get(
                    where={"source": base_filename}, include=["metadatas", "documents"]
                )
            if (
                all_json_chunks
                and isinstance(all_json_chunks.get("ids"), list)
                and isinstance(all_json_chunks.get("documents"), list)
                and isinstance(all_json_chunks.get("metadatas"), list)
            ):
                for i, doc_id in enumerate(all_json_chunks["ids"]):
                    if i < len(all_json_chunks["documents"]) and i < len(
                        all_json_chunks["metadatas"]
                    ):
                        chunk_content = all_json_chunks["documents"][i]
                        chunk_metadata = all_json_chunks["metadatas"][i]
                        if chunk_content is not None and isinstance(
                            chunk_metadata, dict
                        ):
                            documents_for_chroma.append(
                                Document(
                                    page_content=chunk_content, metadata=chunk_metadata
                                )
                            )
            return documents_for_chroma

        # Если файл изменился, продолжаем обработку
        print(
            f"Файл {base_filename} изменился. Инвалидация кэша вопросов/ответов, связанных с '{base_filename}'..."
        )
        vector_db.delete_cached_entries_by_source(base_filename)

        with open(file_path, "r", encoding="utf-8") as f:
            catalog_data = json.load(f)

        current_chroma_ids = set()
        current_chroma_item_paths = set()
        processed_item_paths = set()

        if (
            vector_db.db
            and hasattr(vector_db.db, "_collection")
            and vector_db.db._collection.count() > 0
        ):
            all_json_chunks = vector_db.db.get(
                where={"source": base_filename}, include=["metadatas", "documents"]
            )
            if all_json_chunks and "ids" in all_json_chunks:
                for i, doc_id in enumerate(all_json_chunks["ids"]):
                    if i < len(all_json_chunks["metadatas"]):
                        metadata = all_json_chunks["metadatas"][i]
                        if metadata and "file_path" in metadata:
                            current_chroma_ids.add(doc_id)
                            current_chroma_item_paths.add(metadata["file_path"])

        for i, item in enumerate(catalog_data):
            unique_item_path = f"{file_path}#{i}"
            processed_item_paths.add(unique_item_path)

            item_text_content = json.dumps(item, ensure_ascii=False, sort_keys=True)

            metadata = {
                "source": os.path.basename(file_path),
                "item_name": item.get("name", "N/A"),
                "item_url": item.get("url", "N/A"),
                "index_in_catalog": i,
                "file_path": unique_item_path,
                "file_hash": hashlib.sha256(
                    item_text_content.encode("utf-8")
                ).hexdigest(),
                "file_hash_full": current_file_hash,
                "last_modified": current_last_modified,
                "canonical_content": current_canonical_content,
            }

            existing_chroma_docs = {"ids": [], "metadatas": [], "documents": []}
            if vector_db.db and hasattr(vector_db.db, "get"):
                try:
                    existing_chroma_docs = vector_db.db.get(
                        where={"file_path": unique_item_path},
                        include=["metadatas", "documents"],
                    )
                except Exception as e:
                    print(
                        f"Ошибка при получении данных из ChromaDB для {unique_item_path}: {e}"
                    )
                    existing_chroma_docs = {"ids": [], "metadatas": [], "documents": []}

            is_modified = True
            stored_doc_id = None
            if (
                existing_chroma_docs
                and isinstance(existing_chroma_docs, dict)
                and "ids" in existing_chroma_docs
                and isinstance(existing_chroma_docs["ids"], list)
                and existing_chroma_docs["ids"]
            ):
                stored_doc_id = existing_chroma_docs["ids"][0]
                if (
                    "metadatas" in existing_chroma_docs
                    and isinstance(existing_chroma_docs["metadatas"], list)
                    and existing_chroma_docs["metadatas"]
                ):
                    stored_metadata = existing_chroma_docs["metadatas"][0]
                    if (
                        isinstance(stored_metadata, dict)
                        and "file_hash" in stored_metadata
                        and "last_modified" in metadata
                        and isinstance(metadata["last_modified"], (int, float))
                    ):
                        stored_last_modified = stored_metadata.get("last_modified")
                        if (
                            stored_metadata["file_hash"] == metadata["file_hash"]
                            and isinstance(stored_last_modified, (int, float))
                            and abs(
                                float(stored_last_modified)
                                - float(metadata["last_modified"])
                            )
                            < 60.0
                        ):
                            is_modified = False
                            print(
                                f"Элемент каталога не изменился: {item.get('name', 'N/A')} из {base_filename}"
                            )
                            print(f"Текущий элемент file_hash: {metadata['file_hash']}")
                            print(
                                f"Сохранённый элемент file_hash: {stored_metadata.get('file_hash')}"
                            )
                            chunks = text_splitter.create_documents(
                                [existing_chroma_docs["documents"][0]],
                                metadatas=[stored_metadata],
                            )
                            documents_for_chroma.extend(chunks)

            if is_modified:
                print(
                    f"Обработка нового/измененного элемента каталога: {item.get('name', 'N/A')} из {base_filename}"
                )
                print(f"Текущий file_hash: {metadata['file_hash']}")
                if existing_chroma_docs and existing_chroma_docs["metadatas"]:
                    stored_metadata = (
                        existing_chroma_docs["metadatas"][0]
                        if existing_chroma_docs["metadatas"]
                        else None
                    )
                    if stored_metadata:
                        print(
                            f"Сохранённый file_hash: {stored_metadata.get('file_hash')}"
                        )

                new_chunks_for_chroma, _ = update_document_in_chroma(
                    vector_db,
                    unique_item_path,
                    item_text_content,
                    metadata,
                    metadata["file_hash"],
                    metadata["last_modified"],
                    stored_doc_id,
                )
                documents_for_chroma.extend(new_chunks_for_chroma)

        # Удаляем элементы, которые больше не существуют в JSON
        ids_to_delete_from_json = []
        for doc_id in current_chroma_ids:
            try:
                entry = None
                if vector_db.db and hasattr(vector_db.db, "get"):
                    entry = vector_db.db.get(ids=[doc_id], include=["metadatas"])
                if (
                    entry
                    and isinstance(entry["metadatas"], list)
                    and len(entry["metadatas"]) > 0
                    and isinstance(entry["metadatas"][0], dict)
                    and "file_path" in entry["metадatas"][0]
                ):
                    stored_item_path = entry["metadatas"][0]["file_path"]
                    if (
                        stored_item_path.startswith(file_path + "#")
                        and stored_item_path not in processed_item_paths
                    ):
                        print(
                            f"Обнаружен удаленный элемент из JSON: {stored_item_path}"
                        )
                        ids_to_delete_from_json.append(doc_id)
            except Exception as e:
                print(f"Ошибка при проверке ID {doc_id} в ChromaDB: {e}")

        if ids_to_delete_from_json:
            vector_db.delete_documents(ids_to_delete_from_json)

        return documents_for_chroma

    except Exception as e:
        print(f"Ошибка при обработке файла каталога {file_path}: {str(e)}")
        import traceback

        print(traceback.format_exc())
        return []


def parse_files(directory: str, vector_db: VectorDatabase) -> List[Document]:
    """
    Разбирает файлы из директории и индексирует их в ChromaDB.

    Аргументы:
        directory: Путь к директории, из которой разбирать файлы
        vector_db: Экземпляр VectorDatabase, используемый для индексации

    Возвращает:
        Список разобранных объектов Document

    Генерирует исключения:
        TypeError: Если входные параметры недействительны
        RuntimeError: Если разбор файлов не удается
    """
    if not isinstance(directory, str):
        raise TypeError("directory must be a string")
    if not isinstance(vector_db, VectorDatabase):
        raise TypeError("vector_db must be a VectorDatabase instance")
    if not os.path.isdir(directory):
        raise ValueError(f"Directory does not exist: {directory}")

    all_chunks: List[Document] = []

    # Get list of files in directory
    existing_files_in_data_dir_full_path = {
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    }

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Process JSON files
        if filename.lower().endswith(".json"):
            try:
                json_chunks = process_catalog_data(file_path, vector_db)
                all_chunks.extend(json_chunks)
            except Exception as e:
                print(f"Error processing JSON file {filename}: {e}")
            continue

        # Process PDF, DOCX, TXT files
        if filename.lower().endswith((".pdf", ".docx", ".txt")):
            file_hash = get_file_hash(file_path)
            last_modified = os.path.getmtime(file_path)

            is_modified = True
            stored_doc_id: Optional[str] = None

            # Check if ChromaDB is initialized and has documents
            if not vector_db.db or not hasattr(vector_db.db, "_collection"):
                print("ChromaDB not initialized - processing as new file")
            else:
                collection = vector_db.db._collection
                if not hasattr(collection, "count"):
                    print("ChromaDB collection doesn't support count operation")
                else:
                    count = collection.count() if hasattr(collection, "count") else 0
                    if count > 0:
                        if not hasattr(vector_db.db, "get"):
                            print("ChromaDB doesn't support get operation")
                        else:
                            try:
                                existing_chroma_docs = vector_db.db.get(
                                    where={"source": filename},
                                    include=["metadatas", "documents"],
                                )
                                if (
                                    existing_chroma_docs
                                    and isinstance(existing_chroma_docs, dict)
                                    and "ids" in existing_chroma_docs
                                    and isinstance(existing_chroma_docs["ids"], list)
                                    and existing_chroma_docs["ids"]
                                ):
                                    stored_doc_id = existing_chroma_docs["ids"][0]
                                    if (
                                        "metadatas" in existing_chroma_docs
                                        and isinstance(
                                            existing_chroma_docs["metadatas"], list
                                        )
                                        and existing_chroma_docs["metadatas"]
                                    ):
                                        stored_metadata = existing_chroma_docs[
                                            "metadatas"
                                        ][0]
                                        if isinstance(stored_metadata, dict):
                                            stored_hash = stored_metadata.get(
                                                "file_hash"
                                            )
                                            stored_mtime = stored_metadata.get(
                                                "last_modified"
                                            )

                                            if (
                                                isinstance(stored_hash, str)
                                                and stored_hash == file_hash
                                                and isinstance(
                                                    stored_mtime, (int, float)
                                                )
                                                and abs(
                                                    float(stored_mtime)
                                                    - float(last_modified)
                                                )
                                                < 60.0
                                            ):
                                                is_modified = False
                                                print(
                                                    f"Document unchanged: {filename}."
                                                )

                                                # Restore chunks from database
                                                if (
                                                    "documents" in existing_chroma_docs
                                                    and isinstance(
                                                        existing_chroma_docs[
                                                            "documents"
                                                        ],
                                                        list,
                                                    )
                                                ):
                                                    for i, doc_id in enumerate(
                                                        existing_chroma_docs["ids"]
                                                    ):
                                                        if i < len(
                                                            existing_chroma_docs[
                                                                "documents"
                                                            ]
                                                        ) and i < len(
                                                            existing_chroma_docs[
                                                                "metadatas"
                                                            ]
                                                        ):
                                                            chunk_content = (
                                                                existing_chroma_docs[
                                                                    "documents"
                                                                ][i]
                                                            )
                                                            chunk_metadata = (
                                                                existing_chroma_docs[
                                                                    "metadatas"
                                                                ][i]
                                                            )
                                                            if (
                                                                chunk_content
                                                                is not None
                                                                and isinstance(
                                                                    chunk_metadata, dict
                                                                )
                                                            ):
                                                                all_chunks.append(
                                                                    Document(
                                                                        page_content=chunk_content,
                                                                        metadata=chunk_metadata,
                                                                    )
                                                                )
                            except Exception as e:
                                print(
                                    f"Error checking existing documents for {filename}: {e}"
                                )

            # Обрабатываем только если файл был изменён
            if is_modified:
                print(f"Обработка нового/измененного документа: {filename}.")

                # Инвалидация кэша по источнику
                vector_db.delete_cached_entries_by_source(filename)

                try:
                    elements = partition(filename=file_path)
                    full_text_from_file = "\n\n".join(
                        [e.text for e in elements if e.text and e.text.strip()]
                    )

                    if not full_text_from_file.strip():
                        print(
                            f"Внимание: Не удалось извлечь текст из {filename}. Пропускаем."
                        )
                        continue

                    file_metadata = {
                        "source": filename,
                        "file_path": file_path,
                        "file_hash": file_hash,
                        "last_modified": last_modified,
                    }

                    new_chunks_for_chroma, _ = update_document_in_chroma(
                        vector_db,
                        file_path,
                        full_text_from_file,
                        file_metadata,
                        file_hash,
                        last_modified,
                        stored_doc_id=stored_doc_id,
                    )
                    all_chunks.extend(new_chunks_for_chroma)

                except Exception as e:
                    print(f"Ошибка при обработке файла {file_path}: {e}")
                    import traceback

                    traceback.print_exc()

    return all_chunks


def cleanup_deleted_files(vector_db: VectorDatabase, input_dir: str) -> None:
    """
    Проверяет на удаленные файлы и удаляет их записи из ChromaDB.

    Аргументы:
        vector_db: Экземпляр VectorDatabase для проверки
        input_dir: Директория для проверки на существование файлов

    Generators:
        TypeError: Если аргументы переданы некорректно
        RuntimeError: Если операции с ChromaDB fail
    """
    if not isinstance(vector_db, VectorDatabase):
        raise TypeError("vector_db must be a VectorDatabase instance")
    if not isinstance(input_dir, str):
        raise TypeError("input_dir must be a string")
    if not os.path.isdir(input_dir):
        raise ValueError(f"input_dir does not exist: {input_dir}")

    print("\n--- Проверка на удаленные файлы ---")
    existing_base_filenames_in_data_dir = set()
    for filename in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, filename)):
            existing_base_filenames_in_data_dir.add(filename)

    indexed_sources = set()

    # Validate ChromaDB state
    if not vector_db.db:
        print("Основной индекс ChromaDB не инициализирован. Пропускаем проверку.")
        print("--- Проверка на удаленные файлы завершена ---")
        return

    try:
        # Check collection exists and is accessible
        if not hasattr(vector_db.db, "_collection"):
            print("Коллекция ChromaDB не доступна.")
            return

        collection = vector_db.db._collection
        if not hasattr(collection, "count"):
            print("Коллекция ChromaDB не поддерживает операцию count")
            return

        # Get count safely
        count = collection.count() if hasattr(collection, "count") else 0
        if count == 0:
            print("Основной индекс ChromaDB пуст. Пропускаем проверку.")
            print("--- Проверка на удаленные файлы завершена ---")
            return

        if not hasattr(collection, "get"):
            print("Коллекция ChromaDB не поддерживает операцию get")
            return

        # Get all items with type checking
        all_items_in_db = collection.get(include=["metadatas"])
        if not all_items_in_db or not isinstance(all_items_in_db, dict):
            print("Не удалось получить данные из ChromaDB.")
            return

        metadatas = all_items_in_db.get("metadatas", [])
        if not isinstance(metadatas, list):
            print("Неверный формат метаданных в ChromaDB.")
            return

        # Collect indexed sources
        for metadata in metadatas:
            if not isinstance(metadata, dict):
                continue
            source = metadata.get("source")
            if source and isinstance(source, str):
                indexed_sources.add(source)

        # Process deleted files
        for indexed_source_name in indexed_sources:
            if indexed_source_name not in existing_base_filenames_in_data_dir:
                print(f"Обнаружен удаленный файл: '{indexed_source_name}'")

                if not hasattr(vector_db.db, "get"):
                    print("ChromaDB не поддерживает операцию get")
                    continue

                # Get documents to delete
                result = vector_db.db.get(where={"source": indexed_source_name})
                if not result or not isinstance(result, dict):
                    continue

                ids_to_delete = result.get("ids", [])
                if not isinstance(ids_to_delete, list):
                    continue

                if ids_to_delete:
                    try:
                        vector_db.delete_documents(ids_to_delete)
                        print(
                            f"Удалено {len(ids_to_delete)} записей из ChromaDB для '{indexed_source_name}'."
                        )
                    except Exception as e:
                        print(f"Ошибка при удалении документов: {e}")

                # Clean cache entries
                try:
                    vector_db.delete_cached_entries_by_source(indexed_source_name)
                except Exception as e:
                    print(f"Ошибка при очистке кэша: {e}")

    except Exception as e:
        raise RuntimeError(f"Ошибка при проверке удаленных файлов: {e}")

    print("--- Проверка на удаленные файлы завершена ---")


class RAGSystem:
    """
    Основной класс системы Retrieval Augmented Generation (RAG).
    Оркестрирует работу с хранилищем документов, векторной базой и LLM для ответа на вопросы.
    """

    def __init__(self, config: Config) -> None:
        """
        Инициализирует систему RAG.

        Args:
            config (Config): Объект конфигурации системы.

        Raises:
            ValueError: Если конфигурация невалидна
            TypeError: Если параметры конфигурации неверных типов
        """
        if not isinstance(config, Config):
            raise TypeError("config must be a Config instance")
        if not hasattr(config, "CHROMA_DB_PATH") or not hasattr(
            config, "CHROMA_CACHE_PATH"
        ):
            raise ValueError("Invalid config - missing required paths")

        self.config = config
        self.vector_db = VectorDatabase(
            config.CHROMA_DB_PATH, config.CHROMA_CACHE_PATH, embeddings
        )
        self.llm: ChatOpenAI = llm
        self.retriever: Optional[Any] = None
        self.qa_chain: Optional[LLMChain] = None
        self.prompt: Optional[PromptTemplate] = None

        # Initialize with empty prompt template as fallback
        self.prompt = PromptTemplate.from_template("Вопрос: {input}\nОтвет: ")

    def initialize(self):
        """
        Инициализирует компоненты системы: загружает/обрабатывает документы и настраивает цепочки LangChain.
        """
        print("Инициализация RAG системы...")
        self.vector_db.load_or_create()
        self.vector_db.load_or_create_cache()
        cleanup_deleted_files(self.vector_db, self.config.INPUT_DIR)
        documents = parse_files(self.config.INPUT_DIR, self.vector_db)

        num_docs_in_chroma = 0
        if self.vector_db.db and hasattr(self.vector_db.db, "_collection"):
            num_docs_in_chroma = self.vector_db.db._collection.count()

        if num_docs_in_chroma == 0:
            print(
                "ВНИМАНИЕ: ChromaDB индекс пуст. RAG система может работать неоптимально."
            )
        else:
            print(
                f"Документы загружены и векторная база инициализирована. Индекс содержит {num_docs_in_chroma} элементов."
            )
        self._init_chains()
        print("Цепочки LangChain инициализированы.")

    def _init_chains(self):
        """Инициализирует промпт и цепочки LangChain для RAG."""
        try:
            prompt_template = """
            Ты — умный ассистент, специализирующийся на ювелирных украшениях.
            Ваши основные задачи:
            1. Отвечать на вопросы о ювелирных украшениях, их характеристиках, использовании и ценах по следующему контексту: {context}
            2. Помогать клиентам в выборе подходящих товаров.
            Ваша цель — предоставлять полезные, понятные и дружелюбные ответы.
            Если вы не знаешь ответа, просто скажи: «Я не знаешь». Не придумывай информацию.
            При предложении товаров старайся быть конкретным и описывать, как товар может помочь.
            Если для ответа требуется больше информации, задавай уточняющие вопросы.
            Вопрос: {input}
            """
            self.prompt = PromptTemplate(
                input_variables=["context", "input"],
                template=prompt_template,
            )

            num_docs_in_chroma = 0
            if (
                self.vector_db.db
                and hasattr(self.vector_db.db, "_collection")
                and hasattr(self.vector_db.db._collection, "count")
            ):
                num_docs_in_chroma = self.vector_db.db._collection.count()

            if (
                self.vector_db.db is not None
                and num_docs_in_chroma > 0
                and hasattr(self.vector_db.db, "as_retriever")
            ):
                self.retriever = self.vector_db.db.as_retriever(
                    search_type="similarity", search_kwargs={"k": 4}
                )
                from langchain.chains import create_retrieval_chain
                from langchain.chains.combine_documents import (
                    create_stuff_documents_chain,
                )

                combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
                self.qa_chain = create_retrieval_chain(
                    self.retriever, combine_docs_chain
                )
            else:
                print(
                    "ВНИМАНИЕ: Основной ChromaDB индекс пуст или недоступен. Будет использоваться простая LLM-цепочка."
                )
                self.qa_chain = LLMChain(
                    llm=self.llm,
                    prompt=PromptTemplate(
                        input_variables=["input"], template="Вопрос: {input}\nОтвет: "
                    ),
                )
                self.retriever = None
        except Exception as e:
            print(f"Ошибка при инициализации цепочек: {e}")
            # Fallback to simple LLM chain
            self.qa_chain = LLMChain(
                llm=self.llm,
                prompt=PromptTemplate(
                    input_variables=["input"], template="Вопрос: {input}\nОтвет: "
                ),
            )
            self.retriever = None

    def query(self, question: str, use_cache: bool = True) -> str:
        """
        Обрабатывает запрос пользователя с помощью системы RAG и кэша.

        Аргументы:
            question: Вопрос, который нужно ответить
            use_cache: Использовать семантический кэш

        Возвращает:
            str: Сгенерированный ответ

        Генерирует исключения:
            ValueError: Если вопрос невалидный
            RuntimeError: Если обработка запроса не удалась
        """
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")

        print(f"\n--- Обработка запроса: {question} ---")

        # Тry cache first if enabled
        if use_cache and self.vector_db:
            try:
                cached_answer = self.vector_db.get_cached_answer(question)
                if cached_answer:
                    print("Ответ найден в кэше.")
                    return str(cached_answer)
                print("Кэш не дал совпадений. Обращаюсь к модели...")
            except Exception as e:
                print(f"Ошибка при проверке кэша: {e}")

        # Check if we have documents in ChromaDB
        num_docs_in_chroma = 0
        if (
            self.vector_db
            and self.vector_db.db
            and hasattr(self.vector_db.db, "_collection")
            and hasattr(self.vector_db.db._collection, "count")
        ):
            num_docs_in_chroma = self.vector_db.db._collection.count()

        # Fallback to simple LLM if no documents or retriever
        if not self.retriever or num_docs_in_chroma == 0:
            print("Используется простая LLM-цепочка (нет документов для ретривера).")
            try:
                result = self.llm.invoke(f"Вопрос: {question}\nОтвет: ")
                answer = str(
                    result.content if hasattr(result, "content") else str(result)
                )
                print("Ответ сгенерирован LLM без использования документов.")
                if use_cache and self.vector_db:
                    self.vector_db.add_to_cache(question, answer, sources=[])
                    print("Ответ добавлен в кэш.")
                return answer
            except Exception as e:
                raise RuntimeError(f"Ошибка при генерации ответа: {e}")

        # Use RAG with documents
        try:
            if not self.qa_chain:
                raise RuntimeError("QA chain not initialized")

            result = self.qa_chain.invoke({"input": question})
            if not result or not isinstance(result, dict):
                raise RuntimeError("Invalid QA chain response")

            answer = str(result.get("answer", "Не удалось сгенерировать ответ."))
            retrieved_sources = []

            if "context" in result and isinstance(result["context"], list):
                retrieved_docs = result["context"]
                print(f"Найдено {len(retrieved_docs)} релевантных документов.")

                for i, doc in enumerate(retrieved_docs):
                    if not hasattr(doc, "metadata") or not hasattr(doc, "page_content"):
                        continue

                    metadata = doc.metadata if hasattr(doc, "metadata") else {}
                    source = metadata.get("source", "N/A")
                    if ":" in source:
                        base_source = os.path.basename(source.split("#")[0])
                    else:
                        base_source = os.path.basename(source)

                    if base_source not in retrieved_sources:
                        retrieved_sources.append(base_source)

                    print(
                        f"--- Документ {i+1} (Источник: {metadata.get('source', 'N/A')}, "
                        f"Item: {metadata.get('item_name', 'N/A')}) ---"
                    )
                    print(f"Содержимое (часть): {doc.page_content[:500]}...")
                    print("---------------------------------------")

            print("Ответ сгенерирован LLM с использованием документов.")
            if use_cache and self.vector_db:
                self.vector_db.add_to_cache(question, answer, sources=retrieved_sources)
                print("Ответ добавлен в кэш.")
            return answer

        except Exception as e:
            print(f"Ошибка при обработке запроса: {e}")
            # Fallback to simple LLM on error
            try:
                result = self.llm.invoke(f"Вопрос: {question}\nОтвет: ")
                return str(
                    result.content if hasattr(result, "content") else str(result)
                )
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Ошибка при генерации ответа: {fallback_error}"
                ) from e

    def close(self):
        """Закрывает соединения с базой данных."""
        self.vector_db.db = None
        self.vector_db.cache_db = None
        gc.collect()
        print("Ссылки на базы данных ChromaDB очищены.")


def clean_data(vector_db: Optional[VectorDatabase] = None):
    """Очищает существующие индексы ChromaDB."""
    print("\n--- Очистка данных ---")
    print("Выполняется очистка старых индексов ChromaDB...")

    # Закрываем соединения, если vector_db передаётся
    if vector_db:
        vector_db.db = None
        vector_db.cache_db = None
        print("Соединения с ChromaDB закрыты.")
        gc.collect()

    if os.path.exists(config.CHROMA_DB_PATH):
        try:
            shutil.rmtree(config.CHROMA_DB_PATH)
            print(f"Удалена папка основного индекса ChromaDB: {config.CHROMA_DB_PATH}")
        except PermissionError as e:
            print(
                f"Ошибка при удалении основного индекса: {e}. Возможно, файлы заняты."
            )

    if os.path.exists(config.CHROMA_CACHE_PATH):
        try:
            shutil.rmtree(config.CHROMA_CACHE_PATH)
            print(f"Удалена папка кэша ChromaDB: {config.CHROMA_CACHE_PATH}")
        except PermissionError as e:
            print(f"Ошибка при удалении кэша: {e}. Возможно, файлы заняты.")

    print("Очистка завершена.")
    input("\nНажмите Enter для продолжения...")


def clear_semantic_cache(vector_db: VectorDatabase):
    """Полностью очищает семантический кэш, удаляя все записи из коллекции."""
    print("\n--- Очистка семантического кэша ---")

    # Инициализация кэша, если он ещё не загружен
    if vector_db.cache_db is None:
        try:
            vector_db.load_or_create_cache()
            print("Коллекция кэша ChromaDB инициализирована.")
        except Exception as e:
            print(f"Ошибка при инициализации кэша: {e}")
            print("Семантический кэш не очищен.")
            input("\nНажмите Enter для продолжения.")
            return

    try:
        # Очищаем все записи из кэша
        if (
            vector_db.cache_db
            and hasattr(vector_db.cache_db, "_collection")
            and vector_db.cache_db._collection.count() > 0
        ):
            all_ids = vector_db.cache_db._collection.get()["ids"]
            if all_ids:
                vector_db.cache_db.delete(ids=all_ids)
                print(f"Удалено {len(all_ids)} записей из семантического кэша.")
            else:
                print("Семантический кэш пуст, записей для удаления нет.")
        else:
            print("Семантический кэш пуст, коллекция не содержит записей.")

        # Вместо ручного сброса коллекции просто закрываем соединение
        del vector_db.cache_db
        vector_db.cache_db = None
        gc.collect()
        print("Соединение с кэшем ChromaDB закрыто.")
    except Exception as e:
        print(f"Ошибка при очистке кэша: {e}")
        print("Семантический кэш не очищен.")
    else:
        print("Семантический кэш очищен.")

    input("\nНажмите Enter для продолжения...")


def run_indexing(vector_db: VectorDatabase):
    """Запускает процесс индексации документов."""
    print("\n--- Запуск индексации документов ---")
    rag_system = RAGSystem(config)
    rag_system.vector_db = vector_db  # Используем переданный vector_db
    rag_system.initialize()
    print("Индексация завершена.")
    rag_system.close()
    input("\nНажмите Enter для продолжения...")


def run_interactive_chat(vector_db: VectorDatabase):
    """Запускает интерактивный режим чата."""
    print("\n--- Интерактивный чат ---")
    print("Запуск интерактивного чата. Чтобы выйти, введите 'выход' или 'exit'.")
    rag_system = RAGSystem(config)
    rag_system.vector_db = vector_db  # Используем переданный vector_db
    rag_system.initialize()
    while True:
        question = input("\nВаш вопрос: ")
        if question.lower() in ("выход", "exit"):
            break
        answer = rag_system.query(question)
        print(f"Ответ: {answer}")
    rag_system.close()
    del rag_system
    gc.collect()
    print("Интерактивный чат завершён.")
    input("\nНажмите Enter для продолжения...")


def run_test_questions(vector_db: VectorDatabase):
    """Запускает прогон с предопределенными тестовыми вопросами."""
    print("\n--- Прогон тестовых вопросов ---")
    rag_system = RAGSystem(config)
    rag_system.vector_db = vector_db  # Используем переданный vector_db
    rag_system.initialize()
    questions = [
        "Какие серьги подойдут для вечернего мероприятия?",
        "Посоветуйте украшения для свадьбы",
        "Какие есть золотые кольца?",
        "Сколько стоит Колье с сапфирами?",
        "Как ухаживать за жемчугом?",
        "Опишите Серебряные серьги с аметистом.",
        "Что такое Печатка с ониксом?",
        "Какие украшения есть для мужчин?",
        "Есть ли у вас что-то с бриллиантами?",
    ]
    for i, question in enumerate(questions):
        print(f"\n--- Тестовый вопрос {i+1}/{len(questions)} ---")
        answer = rag_system.query(question)
        print(f"Ответ: {answer}\n")
        if i < len(questions) - 1:
            input("Нажмите Enter, чтобы перейти к следующему вопросу...")
    print("Прогон тестовых вопросов завершён.")
    rag_system.close()
    input("\nНажмите Enter для продолжения...")


def display_menu():
    """Отображает главное меню приложения."""
    print("\n" + "=" * 50)
    print("          Система RAG для ювелирных украшений")
    print("=" * 50)
    print("1. Очистить данные (индексы ChromaDB)")
    print("2. Проиндексировать документы (из папки 'data')")
    print("3. Запустить интерактивный чат")
    print("4. Запустить тестовые вопросы")
    print("5. Очистить семантический кэш (полностью)")
    print("0. Выйти")
    print("=" * 50)


def main():
    """Главная функция консольного приложения."""
    # Создаём единый экземпляр VectorDatabase
    vector_db = VectorDatabase(
        config.CHROMA_DB_PATH, config.CHROMA_CACHE_PATH, embeddings
    )

    while True:
        display_menu()
        choice = input("Выберите опцию: ")
        if choice == "1":
            clean_data(vector_db)
        elif choice == "2":
            run_indexing(vector_db)  # Передаём vector_db
        elif choice == "3":
            run_interactive_chat(vector_db)  # Передаём vector_db
        elif choice == "4":
            run_test_questions(vector_db)  # Передаём vector_db
        elif choice == "5":
            clear_semantic_cache(vector_db)  # Передаём vector_db
        elif choice == "0":
            print("Выход из приложения.")
            # Закрываем соединения перед выходом
            vector_db.db = None
            vector_db.cache_db = None
            print("Соединения с ChromaDB закрыты.")
            break
        else:
            print("Неверный ввод. Пожалуйста, выберите число от 0 до 5.")


if __name__ == "__main__":
    main()

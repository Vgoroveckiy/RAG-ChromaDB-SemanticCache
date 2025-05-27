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
llm = ChatOpenAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)

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
    ):
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

    def load_or_create(self, force_recreate: bool = False):
        """
        Загружает или создает основную коллекцию ChromaDB.
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

        if self.db._collection.count() > 0:
            print(
                f"Основная коллекция ChromaDB 'documents' загружена из {self.db_path}."
            )
        else:
            print(f"Новая основная коллекция ChromaDB создана в {self.db_path}.")

    def load_or_create_cache(self, force_recreate: bool = False):
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
        if self.cache_db._collection.count() > 0:
            print(f"Переменная collection_name изменена на 'current'.")
        else:
            print(f"Переменная collection_name создана в {self.cache_path}.")

    def add_to_cache(
        self, question: str, answer: str, sources: Optional[List[str]] = None
    ):
        """
        Добавляет вопрос и соответствующий ответ в кэш ChromaDB.
        """
        if not self.cache_db:
            self.load_or_create_cache()

        # Очистка кэша перед добавлением новой записи
        self.cleanup_expired_cache_entries(config.CACHE_TTL_DAYS)

        doc_id = str(uuid.uuid4())
        metadata = {
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "sources": json.dumps(sources if sources is not None else []),
            "doc_id": doc_id,
        }
        try:
            self.cache_db.add_texts(
                texts=[question],
                metadatas=[metadata],
                ids=[doc_id],
            )
            print(f"Добавлена запись в кэш с ID: {doc_id}")
        except Exception as e:
            print(f"Ошибка при добавлении записи в кэш ChromaDB: {e}")

    def get_cached_answer(
        self,
        question: str,
        similarity_threshold: float = 0.1,
    ) -> Optional[str]:
        """
        Пытается найти кэшированный ответ на вопрос, если существует достаточно похожий запрос.
        """
        if not self.cache_db:
            self.load_or_create_cache()

        if self.cache_db._collection.count() == 0:
            return None  # Кэш пуст, не тратим время на поиск

        try:
            results = self.cache_db.similarity_search_with_score(query=question, k=1)
        except Exception as e:
            print(f"Ошибка при поиске в кэше: {e}")
            return None

        if not results:
            return None  # Нет результатов

        doc, score = results[0]

        if score <= similarity_threshold:
            print(f"Кэш: Найдено совпадение с расстоянием L2 = {score:.4f}")
            return doc.metadata["answer"]
        else:
            print(
                f"Кэш: Ближайшее совпадение с расстоянием L2 = {score:.4f} (выше порога {similarity_threshold})"
            )
            return None

    def delete_documents(self, doc_ids: List[str]):
        """
        Удаляет документы из ChromaDB по их внутренним ID.
        """
        if not self.db:
            print("ChromaDB не инициализирована. Невозможно удалить документы.")
            return

        if doc_ids:
            try:
                self.db.delete(ids=doc_ids)
                print(f"Удалено {len(doc_ids)} старых чанков из ChromaDB.")
            except Exception as e:
                print(f"Ошибка при удалении чанков из ChromaDB: {e}")
        else:
            print("Нет ID для удаления из ChromaDB.")

    def delete_cached_entries_by_source(self, source_file_name: str):
        if not self.cache_db:
            print("Кэш ChromaDB не инициализирован.")
            return

        if self.cache_db._collection.count() == 0:
            print("Кэш пуст.")
            return

        try:
            all_entries = self.cache_db._collection.get(include=["metadatas"])
        except Exception as e:
            print(f"Ошибка при получении записей из кэша: {e}")
            return

        ids_to_delete = []

        if "ids" in all_entries and "metadatas" in all_entries:
            for i, doc_id in enumerate(all_entries["ids"]):
                if i < len(all_entries["metadatas"]):
                    metadata = all_entries["metadatas"][i]
                    if metadata and metadata.get("sources"):
                        try:
                            sources = json.loads(metadata["sources"])
                            if source_file_name in sources:
                                ids_to_delete.append(doc_id)
                        except json.JSONDecodeError as e:
                            print(f"Ошибка десериализации 'sources' для {doc_id}: {e}")

        if ids_to_delete:
            try:
                self.cache_db.delete(ids=ids_to_delete)
                print(
                    f"Удалено {len(ids_to_delete)} записей, связанных с '{source_file_name}'."
                )
            except Exception as e:
                print(f"Ошибка при удалении записей: {e}")
        else:
            print(f"Не найдено записей, связанных с '{source_file_name}'.")

    def cleanup_expired_cache_entries(self, ttl_days: float):
        """
        Удаляет записи из семантического кэша, срок жизни которых истек (старше ttl_days).
        """
        if not self.cache_db:
            print("Кэш ChromaDB не инициализирован. Пропускаем очистку.")
            return

        print(
            f"--- Очистка семантического кэша (записи старше {ttl_days:.8f} дней) ---"
        )

        if self.cache_db._collection.count() == 0:
            print("Кэш пуст, нет просроченных записей для очистки.")
            print("--- Очистка кэша завершена ---")
            return

        current_time = datetime.now()
        expiration_threshold = current_time - timedelta(days=ttl_days)
        ids_to_delete = []
        all_entries = self.cache_db._collection.get(include=["metadatas"])

        if all_entries and "ids" in all_entries:
            for i, doc_id in enumerate(all_entries["ids"]):
                metadata = all_entries["metadatas"][i]
                if metadata and metadata.get("timestamp"):
                    try:
                        cache_timestamp_str = metadata["timestamp"]
                        cache_datetime = datetime.fromisoformat(cache_timestamp_str)
                        if cache_datetime < expiration_threshold:
                            ids_to_delete.append(doc_id)
                    except ValueError as e:
                        print(
                            f"Предупреждение: Неверный формат timestamp в кэше для {doc_id}: {cache_timestamp_str}. Ошибка: {e}"
                        )

        if ids_to_delete:
            try:
                self.cache_db.delete(ids=ids_to_delete)
                print(
                    f"Удалено {len(ids_to_delete)} просроченных записей из семантического кэша."
                )
            except Exception as e:
                print(f"Ошибка при удалении просроченных записей из кэша: {e}")
        else:
            print("Не найдено просроченных записей в семантическом кэше.")
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
) -> List[Document]:
    """
    Обновляет документ в ChromaDB: удаляет старые чанки и добавляет новые.
    """
    base_file_name_for_cache_invalidation = os.path.basename(file_path.split("#")[0])

    if stored_doc_id:
        try:
            ids_to_delete = []
            if "#" in file_path:
                existing_docs_for_path = vector_db.db.get(
                    where={"file_path": file_path}, include=["metadatas", "documents"]
                )
                if existing_docs_for_path:
                    ids_to_delete.extend(existing_docs_for_path["ids"])
            else:
                existing_docs_for_source = vector_db.db.get(
                    where={"source": base_file_name_for_cache_invalidation},
                    include=["metadatas", "documents"],
                )
                if existing_docs_for_source:
                    ids_to_delete.extend(existing_docs_for_source["ids"])

            if ids_to_delete:
                vector_db.delete_documents(ids_to_delete)
                print(
                    f"Удалены старые ChromaDB чанки для {os.path.basename(file_path)}."
                )
        except Exception as e:
            print(f"Ошибка при попытке удалить старые чанки для {file_path}: {e}")
            pass

    new_chunks = text_splitter.create_documents(
        [full_text_content], metadatas=[metadata]
    )
    new_chroma_ids = []
    if new_chunks:
        texts = [chunk.page_content for chunk in new_chunks]
        metadatas = [chunk.metadata for chunk in new_chunks]
        ids = [str(uuid.uuid4()) for _ in new_chunks]

        if not vector_db.db:
            vector_db.load_or_create()

        vector_db.db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        new_chroma_ids.extend(ids)
        print(f"Добавлено {len(new_chroma_ids)} новых чанков в ChromaDB.")
    else:
        print(f"Нет записей для добавления для {os.path.basename(file_path)}.")

    return new_chunks, new_chroma_ids


def process_catalog_data(file_path: str, vector_db: VectorDatabase) -> List[Document]:
    """
    Читает JSON-файл каталога, преобразует каждую запись в LangChain Document
    и индексирует в ChromaDB. Реализовано точечное обновление.
    """
    documents_for_chroma = []
    try:
        # Читаем файл и вычисляем его хэш
        current_file_hash = get_file_hash(file_path)
        current_last_modified = os.path.getmtime(file_path)
        current_canonical_content = get_canonical_json_content(file_path)
        base_filename = os.path.basename(file_path)

        # Проверяем, есть ли записи для этого файла в ChromaDB
        file_unchanged = False
        existing_file_hash = None
        existing_last_modified = None
        existing_canonical_content = None

        if vector_db.db._collection.count() > 0:
            existing_docs = vector_db.db.get(
                where={"source": base_filename}, include=["metadatas", "documents"]
            )
            if (
                existing_docs
                and "metadatas" in existing_docs
                and existing_docs["metadatas"]
            ):
                # Берем метаданные первого документа для сравнения
                first_metadata = existing_docs["metadatas"][0]
                if first_metadata:  # Проверяем, что метаданные не пустые
                    existing_file_hash = first_metadata.get("file_hash")
                    existing_last_modified = first_metadata.get("last_modified", 0)
                    existing_canonical_content = first_metadata.get(
                        "canonical_content", ""
                    )

                    # Проверяем только содержимое и время модификации
                    if (
                        existing_canonical_content == current_canonical_content
                        and abs(existing_last_modified - current_last_modified) < 60.0
                    ):
                        file_unchanged = True

                # print(f"Текущий file_hash (полный): {current_file_hash}")
                # print(
                #     f"Сохранённый file_hash (полный): {first_metadata.get('file_hash_full', first_metadata.get('file_hash'))}"
                # )
                # print(f"Текущее last_modified: {current_last_modified}")
                # print(f"Сохранённое last_modified: {existing_last_modified}")
                # print(
                #     f"Текущее canonical_content (первые 100 символов): {current_canonical_content[:100]}..."
                # )
                # if existing_canonical_content:
                #     print(
                #         f"Сохранённое canonical_content (первые 100 символов): {existing_canonical_content[:100]}..."
                #     )

        if file_unchanged:
            print(f"Документ не изменился: {base_filename}.")
            # Загружаем существующие документы
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

        if vector_db.db._collection.count() > 0:
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
            if existing_chroma_docs and existing_chroma_docs["ids"]:
                stored_doc_id = existing_chroma_docs["ids"][0]
                if len(existing_chroma_docs["metadatas"]) > 0:
                    stored_metadata = existing_chroma_docs["metadatas"][0]
                    if stored_metadata:
                        if (
                            stored_metadata.get("file_hash") == metadata["file_hash"]
                            and abs(
                                stored_metadata.get("last_modified", 0)
                                - metadata["last_modified"]
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
                entry = vector_db.db.get(ids=[doc_id], include=["metadatas"])
                if (
                    entry
                    and isinstance(entry["metadatas"], list)
                    and len(entry["metadatas"]) > 0
                    and isinstance(entry["metadatas"][0], dict)
                    and "file_path" in entry["metadatas"][0]
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
    Парсит файлы из заданной директории, извлекает полный текст,
    и затем индексирует в ChromaDB. Реализовано точечное обновление.
    """
    all_chunks = []

    # Собираем список всех файлов в директории
    existing_files_in_data_dir_full_path = {
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    }

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Обработка JSON-файлов
        if filename.lower().endswith(".json"):
            json_chunks = process_catalog_data(file_path, vector_db)
            all_chunks.extend(json_chunks)
            continue

        # Обработка PDF, DOCX, TXT
        if filename.lower().endswith((".pdf", ".docx", ".txt")):
            file_hash = get_file_hash(file_path)
            last_modified = os.path.getmtime(file_path)

            is_modified = True
            stored_doc_id = None

            # Проверяем, есть ли уже проиндексированные данные
            if vector_db.db._collection.count() > 0:
                try:
                    existing_chroma_docs = vector_db.db.get(
                        where={"source": filename}, include=["metadatas", "documents"]
                    )
                except Exception as e:
                    print(
                        f"Ошибка при получении данных из ChromaDB для {filename}: {e}"
                    )
                    existing_chroma_docs = {"ids": [], "metadatas": [], "documents": []}

                # Если документ существует — проверяем изменение
                if existing_chroma_docs and existing_chroma_docs["ids"]:
                    stored_doc_id = existing_chroma_docs["ids"][0]
                    if existing_chroma_docs["metadatas"]:
                        stored_metadata = existing_chroma_docs["metadatas"][0]
                        stored_hash = stored_metadata.get("file_hash")
                        stored_mtime = stored_metadata.get("last_modified")

                        if (
                            stored_hash == file_hash
                            and stored_mtime
                            and abs(stored_mtime - last_modified) < 60.0
                        ):
                            is_modified = False
                            print(f"Документ не изменился: {filename}.")
                            # Восстанавливаем чанки из базы
                            for i, doc_id in enumerate(existing_chroma_docs["ids"]):
                                chunk_content = existing_chroma_docs["documents"][i]
                                chunk_metadata = existing_chroma_docs["metadatas"][i]
                                all_chunks.append(
                                    Document(
                                        page_content=chunk_content,
                                        metadata=chunk_metadata,
                                    )
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


def cleanup_deleted_files(vector_db: VectorDatabase, input_dir: str):
    """
    Проверяет соответствие документов в ChromaDB файлам в input_dir.
    Удаляет записи из ChromaDB для файлов, которые больше не существуют.
    """
    print("\n--- Проверка на удаленные файлы ---")
    existing_base_filenames_in_data_dir = set()
    for filename in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, filename)):
            existing_base_filenames_in_data_dir.add(filename)

    indexed_sources = set()
    if vector_db.db._collection.count() > 0:
        all_items_in_db = vector_db.db._collection.get(include=["metadatas"])
        if all_items_in_db and "metadatas" in all_items_in_db:
            for metadata in all_items_in_db["metadatas"]:
                source = metadata.get("source")
                if source:
                    indexed_sources.add(source)
    else:
        print("Основной индекс ChromaDB пуст. Пропускаем проверку.")

    for indexed_source_name in indexed_sources:
        if indexed_source_name not in existing_base_filenames_in_data_dir:
            print(f"Обнаружен удаленный файл: '{indexed_source_name}'")
            ids_to_delete = vector_db.db.get(where={"source": indexed_source_name})[
                "ids"
            ]
            if ids_to_delete:
                vector_db.delete_documents(ids_to_delete)
                print(
                    f"Удалено {len(ids_to_delete)} записей из ChromaDB для '{indexed_source_name}'."
                )
            vector_db.delete_cached_entries_by_source(indexed_source_name)

    print("--- Проверка на удаленные файлы завершена ---")


class RAGSystem:
    """
    Основной класс системы Retrieval Augmented Generation (RAG).
    Оркестрирует работу с хранилищем документов, векторной базой и LLM для ответа на вопросы.
    """

    def __init__(self, config: Config):
        """
        Инициализирует систему RAG.
        Args:
            config (Config): Объект конфигурации системы.
        """
        self.config = config
        self.vector_db = VectorDatabase(
            config.CHROMA_DB_PATH, config.CHROMA_CACHE_PATH, embeddings
        )
        self.llm = llm
        self.retriever: Any = None
        self.qa_chain: Any = None
        self.prompt: PromptTemplate = None

    def initialize(self):
        """
        Инициализирует компоненты системы: загружает/обрабатывает документы и настраивает цепочки LangChain.
        """
        print("Инициализация RAG системы...")
        self.vector_db.load_or_create()
        self.vector_db.load_or_create_cache()
        cleanup_deleted_files(self.vector_db, self.config.INPUT_DIR)
        documents = parse_files(self.config.INPUT_DIR, self.vector_db)
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
        prompt_template = """
        Ты — умный ассистент, специализирующийся на ювелирных украшениях.
        Ваши основные задачи:
        1. Отвечать на вопросы о ювелирных украшениях, их характеристиках, использовании и ценах по следующему контексту: {context}
        2. Помогать клиентам в выборе подходящих товаров.
        Ваша цель — предоставлять полезные, понятные и дружелюбные ответы.
        Если вы не знаешь ответа, просто скажи: «Я не знаю». Не придумывай информацию.
        При предложении товаров старайся быть конкретным и описывать, как товар может помочь.
        Если для ответа требуется больше информации, задавай уточняющие вопросы.
        Вопрос: {input}
        """
        self.prompt = PromptTemplate(
            input_variables=["context", "input"],
            template=prompt_template,
        )
        num_docs_in_chroma = self.vector_db.db._collection.count()
        if self.vector_db.db and num_docs_in_chroma > 0:
            self.retriever = self.vector_db.db.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )
            from langchain.chains import create_retrieval_chain
            from langchain.chains.combine_documents import create_stuff_documents_chain

            combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
            self.qa_chain = create_retrieval_chain(self.retriever, combine_docs_chain)
        else:
            print(
                "ВНИМАНИЕ: Основной ChromaDB индекс пуст. Невозможно инициализировать RetrievalQA. Будет использоваться простая LLM-цепочка."
            )
            self.qa_chain = LLMChain(
                llm=self.llm,
                prompt=PromptTemplate(
                    input_variables=["input"], template="Вопрос: {input}\nОтвет: "
                ),
            )
            self.retriever = None

    def query(self, question: str, use_cache: bool = True) -> str:
        """
        Обрабатывает запрос пользователя, используя RAG-систему и кэш.
        """
        print(f"\n--- Обработка запроса: {question} ---")
        if use_cache:
            cached_answer = self.vector_db.get_cached_answer(question)
            if cached_answer:
                print("Ответ найден в кэше.")
                return cached_answer
            else:
                print("Кэш не дал совпадений. Обращаюсь к модели...")

        num_docs_in_chroma = self.vector_db.db._collection.count()
        if self.retriever is None or num_docs_in_chroma == 0:
            print("Используется простая LLM-цепочка (нет документов для ретривера).")
            result = self.llm.invoke(f"Вопрос: {question}\nОтвет: ")
            answer = result.content if hasattr(result, "content") else str(result)
            print("Ответ сгенерирован LLM без использования документов.")
            if use_cache:
                print(self.vector_db.add_to_cache(question, answer, sources=[]))
                print("Ответ добавлен в кэш.")
            return answer

        result = self.qa_chain.invoke({"input": question})
        answer = result.get("answer", "Не удалось сгенерировать ответ.")
        retrieved_sources = []
        if "context" in result:
            retrieved_docs = result["context"]
            print(f"Найдено {len(retrieved_docs)} релевантных документов.")
            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get("source", "N/A")
                if ":" in source:
                    base_source = os.path.basename(source.split("#")[0])
                else:
                    base_source = os.path.basename(source)
                if base_source not in retrieved_sources:
                    retrieved_sources.append(base_source)
                print(
                    f"--- Документ {i+1} (Источник: {doc.metadata.get('source', 'N/A')}, Item: {doc.metadata.get('item_name', 'N/A')}) ---"
                )
                print(f"Содержимое (часть): {doc.page_content[:500]}...")
                print("---------------------------------------")
        print("Ответ сгенерирован LLM с использованием документов.")
        if use_cache:
            self.vector_db.add_to_cache(question, answer, sources=retrieved_sources)
            print("Ответ добавлен в кэш.")
        return answer

    def close(self):
        """Закрывает соединения с базой данных."""
        self.vector_db.db = None
        self.vector_db.cache_db = None
        gc.collect()
        print("Ссылки на базы данных ChromaDB очищены.")


def clean_data(vector_db: VectorDatabase = None):
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
        if vector_db.cache_db._collection.count() > 0:
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

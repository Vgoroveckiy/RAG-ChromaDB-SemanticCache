import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    Конфигурация системы RAG.

    Определяет пути к файлам и директориям, используемые модели,
    и параметры для обработки и генерации ответов.
    """

    INPUT_DIR: str = "data"
    """Директория для исходных документов, которые будут индексироваться."""

    # ChromaDB сохраняет данные в директории, а не в отдельных файлах, как FAISS
    CHROMA_DB_PATH: str = "./chroma_db"
    """Путь для сохранения основной базы данных ChromaDB, содержащей эмбеддинги документов."""
    CHROMA_CACHE_PATH: str = "./chroma_cache"
    """Путь для сохранения базы данных ChromaDB, используемой для кэширования вопросов и ответов."""

    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    """Название модели эмбеддингов HuggingFace для создания векторных представлений текста."""
    LLM_MODEL: str = "gpt-4o-mini"
    """Название модели LLM (Large Language Model) от OpenAI для генерации ответов."""
    LLM_TEMPERATURE: float = 0.5
    """
    Температура LLM.
    Более высокие значения (например, 0.7) делают ответы более случайными и креативными,
    более низкие значения (например, 0.2) делают их более сфокусированными и детерминированными.
    """
    SUPPORTED_EXTENSIONS: tuple = (".pdf", ".docx", ".txt", ".json")
    """Кортеж поддерживаемых расширений файлов для обработки."""

    # Использование float для TTL для возможности секунд
    CACHE_TTL_DAYS: float = 30  # Устанавливаем TTL на 5 секунд для тестирования
    """Срок жизни записей в семантическом кэше в днях. Записи старше будут удалены."""


# Инициализация объекта конфигурации
config = Config()

# Убедитесь, что OPENAI_API_KEY установлен
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found in .env file. Please create one and add your OPENAI_API_KEY."
    )
os.environ["OPENAI_API_KEY"] = api_key

# Создание директории для входных документов, если она не существует
os.makedirs(config.INPUT_DIR, exist_ok=True)

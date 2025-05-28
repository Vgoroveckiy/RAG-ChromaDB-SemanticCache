import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    Конфигурация системы RAG.

    Определяет пути к файлам и директориям, используемые модели,
    и параметры для обработки и генерации ответов.
    """

    def __init__(self):
        self.INPUT_DIR = "data"
        """
        Директория для исходных документов, которые будут индексироваться.
        ChromaDB сохраняет данные в директории, а не в отдельных файлах, как FAISS
        """
        self.CHROMA_DB_PATH = "./chroma_db"
        """Путь для сохранения основной базы данных ChromaDB, содержащей эмбеддинги документов."""
        self.CHROMA_CACHE_PATH = "./chroma_cache"
        """Путь для сохранения базы данных ChromaDB, используемой для кэширования вопросов и ответов."""

        self.EMBEDDING_MODEL = "BAAI/bge-m3"
        """Название модели эмбеддингов HuggingFace для создания векторных представлений текста."""
        self.LLM_MODEL = "gpt-4o-mini"
        """Название модели LLM (Large Language Model) от OpenAI для генерации ответов."""
        self.LLM_TEMPERATURE = 0.5
        """
        Температура LLM.
        Более высокие значения (например, 0.7) делают ответы более случайными и креативными,
        более низкие значения (например, 0.2) делают их более сфокусированными и детерминированными.
        """
        self.SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".json")
        """Кортеж поддерживаемых расширений файлов для обработки."""
        self.CACHE_TTL_DAYS = 30
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

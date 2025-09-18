import uuid
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import Chroma
from langchain_experimental.open_clip.open_clip import OpenCLIPEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document

def initialize_chroma_client(db_path):
    """
    Инициализирует клиент Chroma DB для хранения векторных представлений.
    
    Параметры:
    db_path (str): Путь к директории базы данных.
    
    Возвращает:
    chromadb.Client: Клиент Chroma DB.
    """
    client = chromadb.PersistentClient(path=db_path)
    embedding_function = OpenCLIPEmbeddingFunction()
    image_loader = ImageLoader()
    
    collection = client.create_collection(
        name='multimodal_collection2',
        embedding_function=embedding_function,
        data_loader=image_loader
    )
    return client

def build_vectorstore(persist_directory):
    """
    Создает векторное хранилище Chroma с функцией встраивания OpenCLIP.
    
    Параметры:
    persist_directory (str): Директория для сохранения данных.
    
    Возвращает:
    Chroma: Объект векторного хранилища.
    """
    vectorstore = Chroma(
        collection_name="mm_rag",
        embedding_function=OpenCLIPEmbeddings(
            model_name="ViT-B-32", 
            checkpoint="laion2b_s34b_b79k"
        ),
        persist_directory=persist_directory
    )
    vectorstore._collection._data_loader = ImageLoader()
    return vectorstore

def build_retriever(vectorstore, docstore, content_storage):
    """
    Создает многофакторный ретривер для поиска по разнотипному контенту.
    
    Параметры:
    vectorstore (Chroma): Векторное хранилище.
    docstore (LocalFileStore): Хранилище документов.
    content_storage (list): Список обработанных элементов контента.
    
    Возвращает:
    MultiVectorRetriever: Сконфигурированный ретривер.
    """
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key,
    )

    def store_document(retriever, item_type, elem, summary_text, ids, start, end, path):
        """
        Добавляет документ и его метаданные в ретривер.
        
        Параметры:
        retriever (MultiVectorRetriever): Ретривер для добавления данных.
        item_type (str): Тип элемента (pdf, image).
        elem: Содержимое элемента.
        summary_text (str): Текстовое описание/резюме.
        ids (str): Уникальный идентификатор документа.
        start (float): Временная метка начала (0 для статичных элементов).
        end (float): Временная метка окончания (0 для статичных элементов).
        path (str): Исходный путь к файлу.
        """
        if not isinstance(summary_text, str):
            summary_text = summary_text[1]
            
        if item_type == 'image':
            id = str(uuid.uuid4())
            elem.save("cur_file.jpeg")
            retriever.vectorstore.add_images(
                ['cur_file.jpeg'], 
                [{'id_key': id, 'start': start, 'end': end, 'path': path}], 
                [ids,]
            )
            id = str(uuid.uuid4())
            retriever.vectorstore.add_documents([
                Document(page_content=summary_text, metadata={'id_key': id, 'start': start, 'end': end, 'path': path})
            ])
        else:
            id = str(uuid.uuid4())
            retriever.vectorstore.add_documents([
                Document(page_content=summary_text, metadata={'id_key': id, 'start': start, 'end': end, 'path': path})
            ])
            
        retriever.docstore.mset([(ids, bytearray(summary_text,'utf-8'))])

    doc_ids = [str(uuid.uuid4()) for _ in content_storage]
    for id, doc in zip(doc_ids, content_storage):
        path = doc['path'] if 'path' in doc else doc['metadata']['path']
        store_document(
            retriever, 
            doc['type'], 
            doc['elem'], 
            doc['sum'], 
            id, 
            doc['metadata']['start'], 
            doc['metadata']['end'], 
            path
        )

    return retriever
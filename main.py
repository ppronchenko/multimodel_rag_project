import os
from config import INPUT_DIR, DB_PATH, VECTOR_DB_PATH, DOCSTORE_PATH
from data_processing.pdf_handler import handle_pdf
from data_processing.image_handler import analyze_image
from storage.vector_store import initialize_chroma_client, build_vectorstore, build_retriever
from retrieval.rag_engine import create_content_summaries, multi_modal_rag
from utils.helpers import get_file_list
from langchain.storage import LocalFileStore

# Global content storage
content_storage = []

def include_pdf(pdf_path):
    """
    Обрабатывает PDF-файл: извлекает содержимое, создает резюме и добавляет в хранилище.
    
    Параметры:
    pdf_path (str): Путь к PDF-файлу.
    """
    table_elements, text_chunks = handle_pdf(pdf_path)
    text_overviews, table_overviews = create_content_summaries(
        text_chunks, table_elements, summarize_texts=True
    )
    
    for item, summary in zip(text_chunks, text_overviews):
        content_storage.append({
            'type': 'pdf',
            'elem': item,
            'sum': summary,
            'path': pdf_path,
            'metadata': {'start':0, 'end':0},
        })
        
    for item, summary in zip(table_elements, table_overviews):
        content_storage.append({
            'type': 'pdf',
            'elem': item,
            'sum': summary,
            'path': pdf_path,
            'metadata': {'start':0, 'end':0},
        })

def include_image(image_path):
    """
    Обрабатывает изображение: генерирует описание и добавляет в хранилище.
    
    Параметры:
    image_path (str): Путь к файлу изображения.
    """
    image_data, image_caption = analyze_image(image_path)
    content_storage.append({
        'type': 'image',
        'elem': image_data,
        'sum': image_caption,
        'path': image_path,
        'metadata': {'start':0, 'end':0},
    })

# Initialize Chroma client
chroma_client = initialize_chroma_client(DB_PATH)

# Process all files in source directory
pdf_list = get_file_list('pdf', "*.pdf")
for file in pdf_list:
    include_pdf(file)

image_list = get_file_list('image', "*.jpg")
for file in image_list:
    include_image(file)

# Build vector store and retriever
vectorstore = build_vectorstore(VECTOR_DB_PATH)
docstore = LocalFileStore(DOCSTORE_PATH)
retriever_multi_vector_img = build_retriever(vectorstore, docstore, content_storage)

# Example queries
if __name__ == "__main__":
    print(multi_modal_rag('Что такое глобальное потепление?', retriever_multi_vector_img))
    print(multi_modal_rag('На кого возлагают основную ответственность за глобальное потепление?', retriever_multi_vector_img))
    print(multi_modal_rag('Как сильно увеличилась темпиратура за последние 20 лет?', retriever_multi_vector_img))
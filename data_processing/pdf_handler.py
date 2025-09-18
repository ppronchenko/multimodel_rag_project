import os
from langchain.text_splitter import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf

def process_pdf_file(pdf_path):
    """
    Извлекает содержимое PDF-файла, включая текст, таблицы и изображения.
    Разбивает текст на фрагменты для дальнейшей обработки.
    
    Параметры:
    pdf_path (str): Путь к PDF-файлу для обработки.
    
    Возвращает:
    list: Список элементов, извлеченных из PDF.
    """
    output_dir = '.'.join(pdf_path.split('.')[:-1])
    return partition_pdf(
        filename=pdf_path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=1500,
        new_after_n_chars=1500,
        combine_text_under_n_chars=500,
        image_output_dir_path=output_dir,
    )

def sort_pdf_content(pdf_content):
    """
    Классифицирует извлеченные элементы PDF на текстовые блоки и таблицы.
    
    Параметры:
    pdf_content (list): Список элементов, извлеченных из PDF.
    
    Возвращает:
    tuple: Два списка - текстовые элементы и таблицы.
    """
    table_elements = []
    text_elements = []
    
    for element in pdf_content:
        if "unstructured.documents.elements.Table" in str(type(element)):
            table_elements.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            text_elements.append(str(element))
            
    return text_elements, table_elements

def handle_pdf(pdf_path):
    """
    Основная функция обработки PDF-файла: извлечение, классификация и разбиение на части.
    
    Параметры:
    pdf_path (str): Путь к PDF-файлу.
    
    Возвращает:
    tuple: Списки таблиц и текстовых фрагментов.
    """
    pdf_content = process_pdf_file(pdf_path)
    text_elements, table_elements = sort_pdf_content(pdf_content)
    
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=100,
    )
    
    combined_text = " ".join(text_elements)
    text_chunks = text_splitter.split_text(combined_text)
    
    return table_elements, text_chunks
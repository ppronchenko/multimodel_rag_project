import streamlit as st
import os
import tempfile
from PIL import Image
import numpy as np
from config import INPUT_DIR, DB_PATH, VECTOR_DB_PATH, DOCSTORE_PATH
from data_processing.pdf_handler import handle_pdf
from data_processing.image_handler import analyze_image
from storage.vector_store import build_vectorstore, build_retriever
from retrieval.rag_engine import create_content_summaries, multi_modal_rag
from langchain.storage import LocalFileStore

# Инициализация сессионного состояния
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.content_storage = []
    st.session_state.retriever = None
    st.session_state.processed_files = set()

def initialize_system():
    """Инициализация системы"""
    if not st.session_state.initialized:
        # Создание директорий если их нет
        os.makedirs(INPUT_DIR, exist_ok=True)
        os.makedirs(DB_PATH, exist_ok=True)
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        os.makedirs(DOCSTORE_PATH, exist_ok=True)
        st.session_state.initialized = True

def process_pdf_file(uploaded_file):
    """Обработка загруженного PDF файла"""
    if uploaded_file.name not in st.session_state.processed_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            table_elements, text_chunks = handle_pdf(tmp_file_path)
            text_overviews, table_overviews = create_content_summaries(
                text_chunks, table_elements, summarize_texts=True
            )
            
            for item, summary in zip(text_chunks, text_overviews):
                st.session_state.content_storage.append({
                    'type': 'pdf',
                    'elem': item,
                    'sum': summary,
                    'path': uploaded_file.name,
                    'metadata': {'start': 0, 'end': 0},
                })
                
            for item, summary in zip(table_elements, table_overviews):
                st.session_state.content_storage.append({
                    'type': 'pdf',
                    'elem': item,
                    'sum': summary,
                    'path': uploaded_file.name,
                    'metadata': {'start': 0, 'end': 0},
                })
                
            st.session_state.processed_files.add(uploaded_file.name)
            os.unlink(tmp_file_path)
            return True
        except Exception as e:
            st.error(f"Ошибка при обработке PDF файла {uploaded_file.name}: {str(e)}")
            os.unlink(tmp_file_path)
            return False
    return True

def process_image_file(uploaded_file):
    """Обработка загруженного изображения"""
    if uploaded_file.name not in st.session_state.processed_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            image_data, image_caption = analyze_image(tmp_file_path)
            st.session_state.content_storage.append({
                'type': 'image',
                'elem': image_data,
                'sum': image_caption,
                'path': uploaded_file.name,
                'metadata': {'start': 0, 'end': 0},
            })
            
            st.session_state.processed_files.add(uploaded_file.name)
            os.unlink(tmp_file_path)
            return True
        except Exception as e:
            st.error(f"Ошибка при обработке изображения {uploaded_file.name}: {str(e)}")
            os.unlink(tmp_file_path)
            return False
    return True

def build_retriever_system():
    """Создание ретривера на основе обработанных файлов"""
    if st.session_state.content_storage and st.session_state.retriever is None:
        try:
            vectorstore = build_vectorstore(VECTOR_DB_PATH)
            docstore = LocalFileStore(DOCSTORE_PATH)
            st.session_state.retriever = build_retriever(
                vectorstore, docstore, st.session_state.content_storage
            )
            return True
        except Exception as e:
            st.error(f"Ошибка при создании системы поиска: {str(e)}")
            return False
    return st.session_state.retriever is not None

def main():
    st.set_page_config(page_title="Multimodal RAG System", page_icon="📚", layout="wide")
    st.title("📚 Multimodal RAG System")
    st.markdown("Система поиска и анализа информации из PDF документов и изображений")
    
    # Инициализация системы
    initialize_system()
    
    # Боковая панель для загрузки файлов
    with st.sidebar:
        st.header("📤 Загрузка файлов")
        st.markdown("Загрузите PDF документы и изображения для анализа")
        
        # Загрузка PDF файлов
        pdf_files = st.file_uploader(
            "Загрузить PDF файлы", 
            type=['pdf'], 
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        # Загрузка изображений
        image_files = st.file_uploader(
            "Загрузить изображения", 
            type=['jpg', 'jpeg', 'png'], 
            accept_multiple_files=True,
            key="image_uploader"
        )
        
        # Кнопка для обработки файлов
        if st.button("🔄 Обработать файлы", type="primary"):
            if pdf_files or image_files:
                progress_bar = st.progress(0)
                total_files = len(pdf_files) + len(image_files)
                processed_count = 0
                
                # Обработка PDF файлов
                for pdf_file in pdf_files:
                    if process_pdf_file(pdf_file):
                        processed_count += 1
                        progress_bar.progress(processed_count / total_files)
                
                # Обработка изображений
                for image_file in image_files:
                    if process_image_file(image_file):
                        processed_count += 1
                        progress_bar.progress(processed_count / total_files)
                
                progress_bar.empty()
                st.success(f"Обработано {processed_count} из {total_files} файлов")
                
                # Создание ретривера
                if build_retriever_system():
                    st.success("Система поиска готова к работе!")
                else:
                    st.warning("Не удалось создать систему поиска")
            else:
                st.warning("Пожалуйста, загрузите файлы для обработки")
        
        # Информация о загруженных файлах
        if st.session_state.processed_files:
            st.divider()
            st.subheader("📊 Обработанные файлы")
            for file_name in st.session_state.processed_files:
                st.markdown(f"- {file_name}")
    
    # Основная область приложения
    if st.session_state.retriever:
        st.header("🔍 Поиск и анализ")
        
        # Тип поиска
        search_type = st.radio(
            "Выберите тип поиска:",
            ["Текстовый запрос", "Поиск по изображению"]
        )
        
        if search_type == "Текстовый запрос":
            query = st.text_input("Введите ваш запрос:", placeholder="Например: Что такое глобальное потепление?")
            
            if st.button("🔎 Найти", type="primary") and query:
                with st.spinner("Поиск информации..."):
                    try:
                        result = multi_modal_rag(query, st.session_state.retriever, is_image=False)
                        st.subheader("Результаты поиска:")
                        st.markdown(result)
                    except Exception as e:
                        st.error(f"Ошибка при поиске: {str(e)}")
                        
        else:  # Поиск по изображению
            image_file = st.file_uploader("Загрузить изображение для поиска", type=['jpg', 'jpeg', 'png'])
            
            if image_file and st.button("🔎 Найти по изображению", type="primary"):
                with st.spinner("Анализ изображения и поиск..."):
                    try:
                        # Сохраняем изображение временно
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            tmp_file.write(image_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        result = multi_modal_rag(tmp_file_path, st.session_state.retriever, is_image=True)
                        st.subheader("Результаты поиска:")
                        st.markdown(result)
                        
                        # Отображаем загруженное изображение
                        st.divider()
                        st.subheader("Анализируемое изображение:")
                        st.image(image_file, caption="Загруженное изображение", use_column_width=True)
                        
                        os.unlink(tmp_file_path)
                    except Exception as e:
                        st.error(f"Ошибка при поиске по изображению: {str(e)}")
    else:
        st.info("📥 Пожалуйста, загрузите файлы и нажмите 'Обработать файлы' для начала работы")
        
        # Демонстрационная информация
        st.divider()
        st.subheader("ℹ️ Как использовать систему:")
        st.markdown("""
        1. **Загрузите файлы** в боковой панели:
           - PDF документы для извлечения текста и таблиц
           - Изображения для генерации описаний
        2. **Нажмите 'Обработать файлы'** для анализа содержимого
        3. **Выберите тип поиска**:
           - Текстовый запрос для поиска по содержимому
           - Поиск по изображению для анализа визуального контента
        4. **Получите точные и структурированные ответы** на основе загруженных материалов
        """)

if __name__ == "__main__":
    main()
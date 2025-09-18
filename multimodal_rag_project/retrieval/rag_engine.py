from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# Initialize LLM
model_path = hf_hub_download(repo_id="unsloth/Qwen3-8B-GGUF", filename="Qwen3-8B-Q6_K.gguf")
llm = Llama(
    model_path,
    n_gpu_layers=-1,
    verbose=False,
    n_ctx=4096
)

def create_content_summaries(texts, tables, summarize_texts=False):
    """
    Создает краткие резюме текстов и таблиц с помощью языковой модели.
    
    Параметры:
    texts (list): Список текстовых элементов.
    tables (list): Список таблиц.
    summarize_texts (bool): Флаг необходимости суммаризации текстов.
    
    Возвращает:
    tuple: Списки резюме текстов и таблиц.
    """
    # Улучшенный промпт для суммаризации текстов
    text_prompt = """Вы являетесь профессиональным редактором и аналитиком контента. Ваша задача - создать точное и лаконичное резюме представленного текста на русском языке для последующего поиска и анализа.

Требования к резюме:
• Сохраняйте фактическую точность и ключевые детали оригинала
• Используйте структурированный подход: выделяйте основные темы, ключевые факты и логические связи
• Если текст уже хорошо структурирован, можете использовать его как основу
• Объем резюме должен быть не более 100 слов
• Отвечайте только на русском языке
• Используйте только информацию из предоставленного текста
• Не добавляйте предположений и домыслов

Если текст слишком короткий или не содержит значимой информации, верните его без изменений."""

    # Улучшенный промпт для суммаризации таблиц
    table_prompt = """Вы являетесь экспертом по анализу структурированных данных. Ваша задача - проанализировать таблицу и создать информативное описание на русском языке для системы поиска.

Алгоритм анализа:
1. Определите тип данных и их структуру
2. Выявите основные тенденции, закономерности и аномалии
3. Отметьте пропущенные значения или необычные элементы
4. Опишите взаимосвязи между различными категориями данных
5. Сформулируйте ключевые выводы в виде четких фактов

Требования:
• Объем описания - не более 1000 символов
• Только русский язык
• Только информация из таблицы
• Без домыслов и предположений
• Если данных недостаточно, укажите это явно"""

    text_overviews = []
    table_overviews = []

    if texts and summarize_texts:
        for text in texts:
            response = llm.create_chat_completion(
                messages = [
                    {"role": "system", "content": text_prompt},
                    {"role": "user", "content": text}
                ]
            )
            summary = response['choices'][0]['message']['content'].split('</think>')[-1].strip()
            text_overviews.append(summary)
    elif texts:
        text_overviews = texts

    if tables:
        for table in tables:
            response = llm.create_chat_completion(
                messages = [
                    {"role": "system", "content": text_prompt},  # Используем тот же промпт для таблиц
                    {"role": "user", "content": table}
                ]
            )
            summary = response['choices'][0]['message']['content'].split('</think>')[-1].strip()
            table_overviews.append(summary)

    return text_overviews, table_overviews

def multi_modal_rag(query, retriever, is_image=False):
    """
    Выполняет поиск и генерацию ответов по запросу пользователя.
    
    Параметры:
    query (str): Текстовый запрос или путь к изображению.
    retriever (MultiVectorRetriever): Ретривер для поиска.
    is_image (bool): Флаг поиска по изображению.
    
    Возвращает:
    str: Сгенерированный ответ.
    """
    if is_image:
        docs = retriever.vectorstore.similarity_search_by_image(query, k=2)
        query = 'Предоставьте краткое содержание'
        print(docs)
    else:
        docs = retriever.vectorstore.search(query, search_type="similarity", k=5)

    # Улучшенный промпт для генерации ответов
    prompt_template = """Вы - профессиональный ассистент, специализирующийся на анализе и систематизации информации. Ваша задача - предоставить точный и содержательный ответ на запрос пользователя на русском языке, используя только предоставленные материалы.

Процесс формирования ответа:
1. Внимательно изучите все предоставленные документы
2. Определите релевантную информацию, относящуюся к запросу
3. Структурируйте ответ логично и последовательно
4. Если в источниках есть противоречивые данные, укажите разные точки зрения
5. При цитировании информации укажите источник

Обязательные требования:
• Отвечайте только на русском языке
• Используйте исключительно информацию из предоставленных материалов
• Не добавляйте внешние знания или предположения
• При недостатке информации, честно укажите это
• Формулируйте мысли ясно и профессионально
• Ответ должен быть по существу вопроса

Предоставленные материалы:
{elements}

Запрос пользователя:
{query}"""
    
    additional_texts = '\n'.join([d.page_content for d in docs])
    
    response = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": prompt_template.format(elements=additional_texts, query=query)},
            {"role": "user", "content": query}
        ]
    )
    
    return response['choices'][0]['message']['content'].split('</think>')[-1].strip()
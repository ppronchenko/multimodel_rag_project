import os
import glob
from config import INPUT_DIR

def get_file_list(directory, extension):
    """
    Получает список файлов с определенным расширением в директории.
    
    Параметры:
    directory (str): Директория для поиска файлов.
    extension (str): Расширение файлов (например, "*.pdf").
    
    Возвращает:
    list: Список путей к файлам.
    """
    dir_path = os.path.join(INPUT_DIR, directory)
    return glob.glob(dir_path + '/' + extension)
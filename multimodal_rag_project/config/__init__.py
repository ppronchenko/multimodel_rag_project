# Configuration settings for the multimodal RAG project
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(PROJECT_ROOT, 'source')
DB_PATH = os.path.join(PROJECT_ROOT, 'db')
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, 'db_v1')
DOCSTORE_PATH = os.path.join(PROJECT_ROOT, 'db_d1')
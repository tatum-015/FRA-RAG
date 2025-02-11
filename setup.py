from setuptools import setup, find_packages

setup(
    name='fra_rag',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'langchain>=0.3.15,<0.4.0',
        'langchain-community>=0.3.0',
        'langchain-openai>=0.3.0',
        'langchain-chroma>=0.2.1,<0.3.0',
        'langchain-huggingface>=0.1.2',
        'langchain-unstructured>=0.1.0',
        'unstructured[pdf-full]>=0.10.25',
        'pdf2image>=1.16.3',
        'pdfminer.six==20240706',
        'pytesseract>=0.3.10',
        'chromadb>=0.4.22',
        'sentence-transformers>=2.2.2',
        'openai>=1.58.1,<2.0.0',
        'tiktoken>=0.7,<1',
        'python-dotenv>=1.0.0',
        'pytest>=7.4.0',
        'nltk>=3.6.0',
        'streamlit>=1.31.0',
        'google-generativeai>=0.8.4'  # Latest version with Flash 2.0 support
    ],
) 
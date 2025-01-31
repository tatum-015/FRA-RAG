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
        # Add any other dependencies here
    ],
) 
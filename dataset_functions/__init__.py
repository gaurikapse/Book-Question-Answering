from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake


def split_documents(file_name: str, org_name: str, dataset_name: str, openai_key: str):
    # Loading PDF document from input_directory
    loader = PyPDFLoader(f"../data/{file_name}")
    doc = loader.load()  # [Document]

    # Embed and upload to vectorstore
    db = DeepLake(
        dataset_path=f"hub://{org_name}/{dataset_name}", embedding_function=OpenAIEmbeddings(openai_api_key=openai_key))
    db.add_documents(doc)

    return dataset_name

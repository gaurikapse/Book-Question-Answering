from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake


def split_documents(file_path: str, org_name: str, dataset_name: str, openai_key: str, deeplake_key: str):
    """
    Loads local pdf files, splits and embeds text and stores embeddings in a DeepLake dataset.
    Inputs:
        file_path—the path to the input file
        org_name—the organization on activeloop in which the dataset is/should be located
        dataset_name—the name of the dataset
        openai_key, deeplake_key: API keys
    """
    # Loading PDF document from input_directory
    loader = PyPDFLoader(file_path)
    doc = loader.load()  # [Document]

    # Embed and upload to vectorstore
    db = DeepLake(
        dataset_path=f"hub://{org_name}/{dataset_name}", embedding_function=OpenAIEmbeddings(openai_api_key=openai_key), token=deeplake_key)
    db.add_documents(doc)

    return dataset_name

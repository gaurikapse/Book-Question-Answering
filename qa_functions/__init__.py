from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

import re
import os


def initialize_retrievers(dataset_name: str, openai_key: str, deeplake_key: str):
    # Initialize LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,
                     openai_api_key=openai_key)

    # Connect to dataset
    db = DeepLake(dataset_path=f"hub://gaurikapse/{dataset_name}",
                  read_only=True, embedding_function=OpenAIEmbeddings(openai_api_key=openai_key), token=deeplake_key)

    # Initialize retriever to retrieve documents based on similar questions
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 2})
    mq_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

    # Setup memory and custom prompt for QA chain
    template = """Use the following extracted parts of a book and the provided chat history to answer the question at the end. 
    Book: {context}
    Chat history: {chat_history}
    Question: {question}
    """
    prompt = PromptTemplate(template=template,
                            input_variables=["context", "chat_history", "question"])
    memory = ConversationBufferMemory(
        memory_key="chat_history", input_key="question")

    # Initialize QA chain
    chain = load_qa_chain(llm=llm, memory=memory, chain_type="map_reduce")

    # Success message
    success = "Success!"
    return mq_retriever, chain, success


def chat(query, mq_retriever, chain, chat_history):
    retrieved_documents = mq_retriever.get_relevant_documents(query=query)
    source = [doc.metadata for doc in retrieved_documents]
    result = chain.run(input_documents=retrieved_documents, question=query)
    chat_history.append([query, result])
    return '', chat_history, source

from dotenv import load_dotenv
import os

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings, AzureChatOpenAI

from typing import Any, Dict, List
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore

INDEX_NAME = "langchain-concepts"


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small",
                                       openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                       azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                       openai_api_version=os.getenv("OPENAI_API_VERSION"))
    docsearch = PineconeVectorStore(embedding=embeddings, index_name=INDEX_NAME)

    # for using openai directly
    # chat = ChatOpenAI(
    #     verbose=True,
    #     temperature=0,
    # )

    # for using azure openai
    chat = AzureChatOpenAI(
        verbose=True,
        temperature=0,
        deployment_name="gpt-35-turbo-16k",
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qa.invoke({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    answer = run_llm("Describe AgentExecutorIterator?")
    print(answer)

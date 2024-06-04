from dotenv import load_dotenv

load_dotenv()

import os

from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings

INDEX_NAME = "langchain-concepts"

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
#                               openai_api_key=os.getenv("OPENAI_API_KEY"),
#                               openai_api_base=os.getenv("OPENAI_API_BASE"))

embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small",
                                   openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                   azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                   openai_api_version=os.getenv("OPENAI_API_VERSION"))


def ingest_docs():
    loader = ReadTheDocsLoader(
        "langchain-docs/python.langchain.com/v0.2/docs"
    )

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()

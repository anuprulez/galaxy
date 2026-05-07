#!/usr/bin/env python3

import argparse
from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

#from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
#from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def search_vector_store(store_path: str, query: str, top_k: int = 5):
    # IMPORTANT: use the same embedding model as during indexing
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.embed_model = embed_model

    storage_context = StorageContext.from_defaults(
        persist_dir=store_path
    )

    index = load_index_from_storage(
        storage_context,
        embed_model=embed_model,
    )

    retriever = index.as_retriever(
        similarity_top_k=top_k
    )

    results = retriever.retrieve(query)

    for i, node_with_score in enumerate(results, start=1):
        node = node_with_score.node
        score = node_with_score.score

        print("=" * 80)
        print(f"Node: {node}")
        print(f"Rank: {i}")
        print(f"Score: {score}")
        print(f"Title: {node.metadata.get('title')}")
        print(f"Type: {node.metadata.get('type')}")
        print(f"Topic: {node.metadata.get('topic', '')}")
        print(f"URL: {node.metadata.get('url', '')}")
        print()
        print(node.get_content()[:10000])
        print(node.metadata)

def search_chromadb_vectors(store_path: str, query: str, top_k: int = 5):

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        base_url="https://api.deepinfra.com/v1/openai",
        model="BAAI/bge-large-en-v1.5",
        api_key="",
        tiktoken_enabled=False,  # Disable tiktoken to avoid DNS issues with custom models
        check_embedding_ctx_length=False  # Disable context length checking
    )

    vectorstore = Chroma(
        persist_directory=store_path,
        collection_name="gtn_tutorials",
        embedding_function=embeddings
    )

    print("Using Chroma vector store for similarity search using query:", query)

    results =  vectorstore.similarity_search(query, k=5)

    print(f"Found {len(results)} similar documents")

    for i, doc in enumerate(results, start=1):
        print("=" * 80)
        print(f"Document: {doc}")
        print(f"Metadata: {doc.metadata}")
        print(f"Metadata: {doc.metadata['source']}")
        print(f"Content: {doc.page_content}")
        print()


    '''def search_vectorstore_db(self,
        query: str,
        limit: int = 5,
    ) -> list[VectorSearchResult]:
        # Initialize embeddings

        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings

        try:

            embeddings = OpenAIEmbeddings(
                base_url="https://api.deepinfra.com/v1/openai",
                model="BAAI/bge-large-en-v1.5",
                api_key="2v7jO5zshabaCb1i0JJQmqVXbRciW5q2",
                tiktoken_enabled=False,
                check_embedding_ctx_length=False
            )

            vectorstore = Chroma(
                persist_directory=GTN_VECTOR_CHROMADB_URL,
                collection_name="gtn_tutorials",
                embedding_function=embeddings
            )

            results =  vectorstore.similarity_search(query, k=limit)

            log.info(f"Found {len(results)} similar documents")

            vector_results = []

            for i, doc in enumerate(results):
                result = VectorSearchResult(
                    page_content=str(doc.page_content),
                    source=doc.metadata.source,
                    content_type=doc.metadata.content_type,
                    data_source=doc.metadata.data_source,
                )
                vector_results.append(result)

            log.info(vector_results)

            return vector_results
        
        except Exception as e:
            log.warning(f"Vector search failed for query '{query}': {e}")
            return []'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("store_path", help="Path to persisted LlamaIndex vector store")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=1)

    args = parser.parse_args()

    '''search_vector_store(
        store_path=args.store_path,
        query=args.query,
        top_k=args.top_k,
    )'''

    search_chromadb_vectors(
        store_path=args.store_path,
        query=args.query,
        top_k=args.top_k,
    )
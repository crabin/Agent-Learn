# 1.导入相关依赖
import os

import dotenv
from openai import APIConnectionError, BadRequestError, NotFoundError
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

dotenv.load_dotenv()


def normalize_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None

    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


# 2.定义文档加载器
loader = TextLoader(file_path='./asset/load/09-ai1.txt',encoding="utf-8")

# 3.加载文档
documents = loader.load()

# 4.定义文本切割器
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# 5.切割文档
docs = text_splitter.split_documents(documents)

# 6.定义嵌入模型
api_key = os.getenv("OPENAI_API_KEY")
base_url = normalize_base_url(os.getenv("OPENAI_API_BASE"))
embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "nomic-embed-text:latest")
using_compatible_endpoint = bool(base_url)

embeddings = OpenAIEmbeddings(
    model=embedding_model,
    openai_api_key=api_key,
    openai_api_base=base_url,
    # Many OpenAI-compatible providers reject token-id arrays for embeddings
    # and only accept raw strings as input.
    check_embedding_ctx_length=not using_compatible_endpoint,
    encoding_format="float",
)

# 获取向量数据库
try:
    db = FAISS.from_documents(documents=docs, embedding=embeddings)
except NotFoundError as exc:
    raise RuntimeError(
        "Embedding request returned 404. "
        f"Resolved base URL: {base_url}. "
        f"Resolved embedding model: {embedding_model}. "
        "This usually means the OpenAI-compatible endpoint does not expose the "
        "embeddings API, or the embedding model name is not supported."
    ) from exc
except APIConnectionError as exc:
    raise RuntimeError(
        "Failed to connect to the embeddings endpoint. "
        f"Resolved base URL: {base_url}. "
        "Please check whether OPENAI_API_BASE points to a reachable host/service."
    ) from exc
except BadRequestError as exc:
    raise RuntimeError(
        "Embedding request returned 400. "
        f"Resolved base URL: {base_url}. "
        f"Resolved embedding model: {embedding_model}. "
        "If you are using an OpenAI-compatible endpoint, make sure it supports "
        "the embeddings API and accepts raw text input."
    ) from exc

# 基于向量数据库获取检索器
retriever = db.as_retriever()

# 进行数据的检索
docs = retriever.invoke(input = "深度学习是什么？")

print(len(docs))

for doc in docs:
    print(f"------{doc}")

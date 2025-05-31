TITLE: Install main LangChain package
DESCRIPTION: Installs the core `langchain` package using either pip or conda. This package provides a starting point but requires separate installation of dependencies for specific integrations.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/how_to/installation.mdx#_snippet_0

LANGUAGE: bash
CODE:
```
pip install langchain
```

LANGUAGE: bash
CODE:
```
conda install langchain -c conda-forge
```

----------------------------------------

TITLE: Invoking LangChain Chat Model with HumanMessage (Python)
DESCRIPTION: Demonstrates how to invoke a LangChain chat model using a list containing a HumanMessage object with text content. This is the standard way to pass user input messages.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/concepts/messages.mdx#_snippet_0

LANGUAGE: python
CODE:
```
from langchain_core.messages import HumanMessage

model.invoke([HumanMessage(content="Hello, how are you?")])
```

----------------------------------------

TITLE: Load, Split PDF, and Add to Vector Store (Python)
DESCRIPTION: Loads text from a PDF file, splits it into smaller chunks using a text splitter, and adds the resulting documents to the vector store.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/vectorstores/cassandra.ipynb#_snippet_20

LANGUAGE: python
CODE:
```
pdf_loader = PyPDFLoader("what-is-philosophy.pdf")
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
docs_from_pdf = pdf_loader.load_and_split(text_splitter=splitter)

print(f"Documents from PDF: {len(docs_from_pdf)}.")
inserted_ids_from_pdf = vstore.add_documents(docs_from_pdf)
print(f"Inserted {len(inserted_ids_from_pdf)} documents.")
```

----------------------------------------

TITLE: Install LangChain with Pip (Bash)
DESCRIPTION: Installs the LangChain library using the pip package manager. This is the standard way to install Python packages. Requires Python and pip to be installed.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/tutorials/llm_chain.ipynb#_snippet_0

LANGUAGE: bash
CODE:
```
pip install langchain
```

----------------------------------------

TITLE: Install LangChain OpenAI Integration Package - Bash
DESCRIPTION: This command installs the necessary Python package to use OpenAI integrations with LangChain. It is the first step required before importing and using any OpenAI-related components.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/providers/openai.mdx#_snippet_0

LANGUAGE: bash
CODE:
```
pip install langchain-openai
```

----------------------------------------

TITLE: Using OllamaLLM for LLMs (Python)
DESCRIPTION: Initializes an `OllamaLLM` instance with a specified model and invokes it with a prompt for text completion. This class provides an interface to Ollama language models within LangChain. Requires the `langchain-ollama` package and a running Ollama server with the specified model.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/libs/partners/ollama/README.md#_snippet_3

LANGUAGE: python
CODE:
```
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3")
llm.invoke("The meaning of life is")
```

----------------------------------------

TITLE: Install LangChain RAG Dependencies (Pip)
DESCRIPTION: Installs the necessary LangChain libraries for building the RAG application using pip, including text splitters, community components, and LangGraph. This command is typically run in a Jupyter or IPython environment.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/tutorials/rag.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
%pip install --quiet --upgrade langchain-text-splitters langchain-community langgraph
```

----------------------------------------

TITLE: Define Tools using @tool Decorator in Python
DESCRIPTION: This Python snippet shows how to define simple tools (add and multiply) using the `@tool` decorator from `langchain_core.tools`. The docstrings are used as tool descriptions.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/how_to/function_calling.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

tools = [add, multiply]
```

----------------------------------------

TITLE: Initialize ChatOpenAI Model (Python)
DESCRIPTION: Initializes a ChatOpenAI model instance using the 'gpt-4o-mini' model name. This sets up the language model for subsequent interactions.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/tutorials/llm_chain.ipynb#_snippet_4

LANGUAGE: python
CODE:
```
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
```

----------------------------------------

TITLE: Initialize and Use Perplexity Chat Model (Python)
DESCRIPTION: This Python snippet demonstrates how to import necessary modules, set the PPLX_API_KEY environment variable (prompting if not set), initialize a Perplexity chat model using LangChain's init_chat_model function, and invoke the model with a simple prompt.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/libs/partners/perplexity/README.md#_snippet_1

LANGUAGE: python
CODE:
```
import getpass
import os

if not os.environ.get("PPLX_API_KEY"):
  os.environ["PPLX_API_KEY"] = getpass.getpass("Enter API key for Perplexity: ")

from langchain.chat_models import init_chat_model

llm = init_chat_model("llama-3.1-sonar-small-128k-online", model_provider="perplexity")
llm.invoke("Hello, world!")
```

----------------------------------------

TITLE: Define, Bind, and Invoke Magic Function Tool
DESCRIPTION: Defines another tool, 'magic_function', using a Pydantic model for input. It binds this tool to the LLM, forces its selection, and then invokes the LLM with a query designed to trigger this tool. The final line shows the resulting AI message.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/chat/llamacpp.ipynb#_snippet_11

LANGUAGE: python
CODE:
```
class MagicFunctionInput(BaseModel):
    magic_function_input: int = Field(description="The input value for magic function")


@tool("get_magic_function", args_schema=MagicFunctionInput)
def magic_function(magic_function_input: int):
    """Get the value of magic function for an input."""
    return magic_function_input + 2


llm_with_tools = llm.bind_tools(
    tools=[magic_function],
    tool_choice={"type": "function", "function": {"name": "get_magic_function"}},
)

ai_msg = llm_with_tools.invoke(
    "What is magic function of 3?",
)

ai_msg
```

----------------------------------------

TITLE: Installing LangChain Dependencies
DESCRIPTION: Commands for installing LangChain using pip or conda package managers.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/tutorials/summarization.ipynb#2025-04-22_snippet_0

LANGUAGE: bash
CODE:
```
pip install langchain
```

LANGUAGE: bash
CODE:
```
conda install langchain -c conda-forge
```

----------------------------------------

TITLE: Configuring and Using CrateDB Semantic Cache with LangChain (Python)
DESCRIPTION: This snippet demonstrates how to initialize and use the CrateDBSemanticCache with LangChain. It requires the 'sqlalchemy', 'langchain', 'langchain-openai', and 'langchain-cratedb' libraries. The code configures OpenAI embeddings, creates a SQLAlchemy engine for CrateDB, sets the global LLM cache, and then invokes a ChatOpenAI model to show the cache in action.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/providers/cratedb.mdx#_snippet_7

LANGUAGE: python
CODE:
```
import sqlalchemy as sa
from langchain.globals import set_llm_cache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_cratedb import CrateDBSemanticCache

# Configure embeddings.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Configure cache.
engine = sa.create_engine("crate://crate@localhost:4200/?schema=testdrive")
set_llm_cache(
    CrateDBSemanticCache(
        embedding=embeddings,
        connection=engine,
        search_threshold=1.0,
    )
)

# Invoke LLM conversation.
llm = ChatOpenAI(model_name="chatgpt-4o-latest")
print()
print("Asking with semantic cache:")
answer = llm.invoke("What is the answer to everything?")
print(answer.content)
```

----------------------------------------

TITLE: Load and Chunk Text Document - Python
DESCRIPTION: Loads a text document from a specified path using `TextLoader`, splits it into smaller chunks based on character count using `CharacterTextSplitter`, and prints the total number of chunks created.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/vectorstores/sap_hanavector.ipynb#_snippet_5

LANGUAGE: Python
CODE:
```
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

text_documents = TextLoader(
    "../../how_to/state_of_the_union.txt", encoding="UTF-8"
).load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
text_chunks = text_splitter.split_documents(text_documents)
print(f"Number of document chunks: {len(text_chunks)}")
```

----------------------------------------

TITLE: Loading and Splitting a PDF Document
DESCRIPTION: Uses the PyPDFLoader from LangChain to load the downloaded PDF file and split its content into individual pages. This prepares the text data for embedding and storage.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/vectorstores/kdbai.ipynb#_snippet_6

LANGUAGE: python
CODE:
```
%%time
print("Read a PDF...")
loader = PyPDFLoader(PDF)
pages = loader.load_and_split()
len(pages)
```

----------------------------------------

TITLE: Load PDF Documents with PyPDFLoader (Python)
DESCRIPTION: Initializes and uses the `PyPDFLoader` to load content from a specified PDF file path into a list of `Document` objects, typically creating one document per page.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/tutorials/retrievers.ipynb#_snippet_5

LANGUAGE: python
CODE:
```
from langchain_community.document_loaders import PyPDFLoader

file_path = "../example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()
```

----------------------------------------

TITLE: Setting OpenAI API Key and Initializing Chat Model
DESCRIPTION: Imports `getpass` and `os` to securely set the OpenAI API key as an environment variable and then initializes a chat model instance from `langchain.chat_models` using the specified model provider and name.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/tools/oxylabs.ipynb#_snippet_7

LANGUAGE: python
CODE:
```
import getpass
import os

from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
```

----------------------------------------

TITLE: Build RAG Chain with DappierRetriever (Python)
DESCRIPTION: Constructs a LangChain RAG (Retrieval Augmented Generation) chain using the DappierRetriever, a prompt template, an LLM, and an output parser, including a helper function to format retrieved documents.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/retrievers/dappier.ipynb#_snippet_6

LANGUAGE: python
CODE:
```
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

----------------------------------------

TITLE: Initializing OpenAI Embeddings Model (Python)
DESCRIPTION: Imports and initializes the `OpenAIEmbeddings` model from `langchain_openai`. This model is used to generate vector representations of text, which are essential for semantic search. Requires the `OPENAI_API_KEY` environment variable.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/cookbook/retrieval_in_sql.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()
```

----------------------------------------

TITLE: Setting up PydanticOutputParser and Prompt in LangChain Python
DESCRIPTION: Defines Pydantic models for structured data, initializes a `PydanticOutputParser` based on one of the models, and creates a `ChatPromptTemplate` that incorporates the parser's format instructions for guiding the LLM output.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/how_to/structured_output.ipynb#_snippet_14

LANGUAGE: Python
CODE:
```
from typing import List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class Person(BaseModel):
    """Information about a person."""

    name: str = Field(..., description="The name of the person")
    height_in_meters: float = Field(
        ..., description="The height of the person expressed in meters."
    )


class People(BaseModel):
    """Identifying information about all people in a text."""

    people: List[Person]


# Set up a parser
parser = PydanticOutputParser(pydantic_object=People)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user query. Wrap the output in `json` tags\n{format_instructions}",
        ),
        ("human", "{query}"),
    ]
).partial(format_instructions=parser.get_format_instructions())
```

----------------------------------------

TITLE: Setup Langchain Knowledge Base from Text File (Python)
DESCRIPTION: Defines a function `setup_knowledge_base` that reads a text file (assumed to be a product catalog), splits the text into chunks using `CharacterTextSplitter`, creates embeddings using `OpenAIEmbeddings`, builds a vector store (`Chroma`) from the texts and embeddings, and finally creates a `RetrievalQA` chain using a `ChatOpenAI` LLM and the vector store retriever. It returns the configured knowledge base chain. Requires Langchain, OpenAI, and Chroma dependencies.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/cookbook/sales_agent_with_context.ipynb#_snippet_10

LANGUAGE: python
CODE:
```
def setup_knowledge_base(product_catalog: str = None):
    """
    We assume that the product knowledge base is simply a text file.
    """
    # load product catalog
    with open(product_catalog, "r") as f:
        product_catalog = f.read()

    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    texts = text_splitter.split_text(product_catalog)

    llm = ChatOpenAI(temperature=0)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(
        texts, embeddings, collection_name="product-knowledge-base"
    )

    knowledge_base = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )
    return knowledge_base
```

----------------------------------------

TITLE: Loading and Splitting Document for Indexing
DESCRIPTION: Imports necessary classes for vector storage (Chroma) and text splitting. It then loads text content from a file (`state_of_the_union.txt`), initializes a CharacterTextSplitter, and splits the document into smaller chunks (`texts`) suitable for indexing in a vector database.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/cookbook/hypothetical_document_embeddings.ipynb#_snippet_10

LANGUAGE: python
CODE:
```
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter

with open("../../state_of_the_union.txt") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)
```

----------------------------------------

TITLE: Storing Document Chunks in Vector Store with Langchain Python
DESCRIPTION: This snippet adds the list of document chunks (`all_splits`) generated by the text splitter into a pre-configured `vector_store`. This process typically involves embedding the chunks and storing the embeddings and text in the vector database. It prints the first three IDs returned by the vector store after adding the documents.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/tutorials/rag.ipynb#_snippet_12

LANGUAGE: python
CODE:
```
document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])
```

----------------------------------------

TITLE: Parse JSON Output with Pydantic (Python)
DESCRIPTION: Demonstrates how to use JsonOutputParser with a Pydantic model to define the expected JSON schema. It sets up a chat model, defines the data structure, creates a parser linked to the Pydantic object, builds a prompt template including format instructions, chains the components, and invokes the chain to get structured output.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/how_to/output_parser_json.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

model = ChatOpenAI(temperature=0)


# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


# And a query intented to prompt a language model to populate the data structure.
joke_query = "Tell me a joke."

# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

chain.invoke({"query": joke_query})
```

----------------------------------------

TITLE: Loading and Splitting Text Data with Embeddings
DESCRIPTION: Loads text content from a file using `TextLoader`, splits it into smaller chunks using `CharacterTextSplitter`, and initializes `OpenAIEmbeddings`. This prepares the text data and the embedding model for creating vector representations.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/vectorstores/timescalevector.ipynb#_snippet_5

LANGUAGE: python
CODE:
```
# Load the text and split it into chunks
loader = TextLoader("../../../extras/modules/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
```

----------------------------------------

TITLE: Configure LangSmith Tracing (Optional)
DESCRIPTION: Optional configuration to enable LangSmith tracing for monitoring model calls by setting environment variables for the API key and tracing flag. This code is commented out by default.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/vectorstores/qdrant.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"
```

----------------------------------------

TITLE: Basic RAG Workflow with Langchain Python
DESCRIPTION: Demonstrates a simple RAG pipeline using Langchain. It defines a system prompt to guide the model, retrieves relevant documents based on a user question, formats the system prompt with the retrieved context, initializes a ChatOpenAI model, and invokes the model with the formatted prompt and the user question to generate an answer grounded in the retrieved information. Requires a pre-configured `retriever` object.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/concepts/rag.mdx#_snippet_0

LANGUAGE: python
CODE:
```
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Define a system prompt that tells the model how to use the retrieved context
system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Context: {context}:"""
    
# Define a question
question = """What are the main components of an LLM-powered autonomous agent system?"""

# Retrieve relevant documents
docs = retriever.invoke(question)

# Combine the documents into a single string
docs_text = "".join(d.page_content for d in docs)

# Populate the system prompt with the retrieved context
system_prompt_fmt = system_prompt.format(context=docs_text)

# Create a model
model = ChatOpenAI(model="gpt-4o", temperature=0) 

# Generate a response
questions = model.invoke([SystemMessage(content=system_prompt_fmt),
                          HumanMessage(content=question)])
```

----------------------------------------

TITLE: Creating Document with Metadata - Langchain - Python
DESCRIPTION: Constructs a Langchain Document object, incorporating both the main text content from the 'text' variable and the contextual information stored in the 'metadata' dictionary. This allows for richer document representation.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/document_loaders/copypaste.ipynb#_snippet_4

LANGUAGE: python
CODE:
```
doc = Document(page_content=text, metadata=metadata)
```

----------------------------------------

TITLE: Load Data, Split, Embed, and Initialize LLM (Python)
DESCRIPTION: Loads data from a web page using WebBaseLoader, splits it into chunks with RecursiveCharacterTextSplitter, creates embeddings using OpenAIEmbeddings, stores them in an in-memory FAISS vector store, and initializes a ChatOpenAI model.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/versions/migrating_chains/retrieval_qa.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
# Load docs
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Store splits
vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# LLM
llm = ChatOpenAI()
```

----------------------------------------

TITLE: Enabling LangChain Debug Mode
DESCRIPTION: Enables the most verbose logging level for LangChain components using `set_debug(True)`, printing all inputs and outputs, including raw data, for comprehensive debugging.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/how_to/debugging.ipynb#_snippet_5

LANGUAGE: python
CODE:
```
from langchain.globals import set_debug

set_debug(True)
set_verbose(False)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_executor.invoke(
    {"input": "Who directed the 2023 film Oppenheimer and what is their age in days?"}
)
```

----------------------------------------

TITLE: Invoke Cohere Chat Model
DESCRIPTION: Demonstrates how to initialize and invoke the Cohere chat model (ChatCohere) with a simple human message.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/providers/cohere.mdx#_snippet_1

LANGUAGE: python
CODE:
```
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
chat = ChatCohere()
messages = [HumanMessage(content="knock knock")]
print(chat.invoke(messages))
```

----------------------------------------

TITLE: Loading, Splitting Documents and Initializing DocArrayInMemorySearch
DESCRIPTION: Loads text documents from a file, splits them into smaller chunks using a character splitter, generates OpenAI embeddings for the chunks, and initializes a `DocArrayInMemorySearch` vector store with the documents and embeddings.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/vectorstores/docarray_in_memory.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
documents = TextLoader("../../how_to/state_of_the_union.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db = DocArrayInMemorySearch.from_documents(docs, embeddings)
```

----------------------------------------

TITLE: Setting OpenAI API Key
DESCRIPTION: Imports `getpass` and `os` to retrieve and set the OpenAI API key as an environment variable, required for using OpenAI embeddings and LLMs.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/retrievers/self_query/chroma_self_query.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
```

----------------------------------------

TITLE: Initialize TileDB Vector Store from Documents (Python)
DESCRIPTION: Loads text documents, splits them into chunks, generates embeddings using HuggingFace, and initializes a TileDB vector store from these documents with a specified index URI and type.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/vectorstores/tiledb.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import TileDB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

raw_documents = TextLoader("../../how_to/state_of_the_union.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
db = TileDB.from_documents(
    documents, embeddings, index_uri="/tmp/tiledb_index", index_type="FLAT"
)
```

----------------------------------------

TITLE: Filter BagelDB Search by Metadata
DESCRIPTION: Illustrates creating a BagelDB cluster from text with associated metadata. It shows how to perform a similarity search and filter the results based on specific metadata values using the `where` parameter. The example finishes by deleting the cluster. Requires providing a list of metadata dictionaries matching the texts.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/vectorstores/bageldb.ipynb#_snippet_4

LANGUAGE: python
CODE:
```
texts = ["hello bagel", "this is langchain"]
metadatas = [{"source": "notion"}, {"source": "google"}]

cluster = Bagel.from_texts(cluster_name="testing", texts=texts, metadatas=metadatas)
cluster.similarity_search_with_score("hello bagel", where={"source": "notion"})
```

LANGUAGE: python
CODE:
```
# delete the cluster
cluster.delete_cluster()
```

----------------------------------------

TITLE: Initialize and Use ChatDeepSeek Model (Python)
DESCRIPTION: Demonstrates how to import the ChatDeepSeek class, initialize a chat model instance specifying the model name, and invoke it with a sample prompt. Requires the DEEPSEEK_API_KEY environment variable to be set.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/libs/partners/deepseek/README.md#_snippet_1

LANGUAGE: Python
CODE:
```
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(model="deepseek-chat")
llm.invoke("Sing a ballad of LangChain.")
```

----------------------------------------

TITLE: Initialize Chat Model (LangChain, Python)
DESCRIPTION: Initializes a ChatOpenAI language model instance with a specific model name ('gpt-4o-mini') for use in the application.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/tutorials/rag.ipynb#_snippet_4

LANGUAGE: python
CODE:
```
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
```

----------------------------------------

TITLE: Creating Custom Runnable Retriever with LangChain Python
DESCRIPTION: This snippet shows how to define a custom retriever using the `@chain` decorator in LangChain. It wraps the `similarity_search` method of a vector store to create a runnable that takes a query string and returns a list of documents. The example then demonstrates using the `batch` method to process multiple queries simultaneously.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/tutorials/retrievers.ipynb#_snippet_17

LANGUAGE: python
CODE:
```
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain


@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)


retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)
```

----------------------------------------

TITLE: Invoking Agent Executor with User Query
DESCRIPTION: Executes the defined agent with the user's query. This triggers the agent to use its tools to find the requested financial data and formulate a response.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/tools/financial_datasets.ipynb#_snippet_9

LANGUAGE: Python
CODE:
```
agent_executor.invoke({"input": query})
```

----------------------------------------

TITLE: Initializing OpenAI or Azure Chat Model (Python)
DESCRIPTION: Demonstrates how to initialize either an OpenAI `ChatOpenAI` model or an Azure `AzureChatOpenAI` model for use in question answering. This model will generate responses based on retrieved information.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/vectorstores/hippo.ipynb#_snippet_7

LANGUAGE: python
CODE:
```
# llm = AzureChatOpenAI(
#     openai_api_base="x x x",
#     openai_api_version="xxx",
#     deployment_name="xxx",
#     openai_api_key="xxx",
#     openai_api_type="azure"
# )

llm = ChatOpenAI(openai_api_key="YOUR OPENAI KEY", model_name="gpt-3.5-turbo-16k")
```

----------------------------------------

TITLE: Load and Split Documents with LangChain
DESCRIPTION: Illustrates how to load text documents using `TextLoader` and split them into smaller chunks using `CharacterTextSplitter`. This prepares the text data for conversion into vector embeddings and insertion into the vector store.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/vectorstores/upstash.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

docs[:3]
```

----------------------------------------

TITLE: Initializing and Running Langchain Agent with Tools - Python
DESCRIPTION: This snippet initializes a Langchain agent using the `serpapi` and `llm-math` tools and a specified LLM. It then runs a query through the agent and flushes a Weights & Biases callback tracker. Requires Langchain, serpapi, and potentially wandb dependencies.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/providers/wandb_tracking.ipynb#_snippet_8

LANGUAGE: Python
CODE:
```
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
agent.run(
    "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?",
    callbacks=callbacks,
)
wandb_callback.flush_tracker(agent, reset=False, finish=True)
```

----------------------------------------

TITLE: Build RAG Chain (LangChain, Python)
DESCRIPTION: Constructs a RAG application graph using LangChain and LangGraph. It includes steps for loading and splitting a web document, indexing chunks into the vector store, defining the application state, and implementing retrieve and generate functions.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/tutorials/rag.ipynb#_snippet_7

LANGUAGE: python
CODE:
```
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
```

----------------------------------------

TITLE: Adding Documents to LangChain Couchbase Vector Store - Python
DESCRIPTION: Demonstrates how to create Document objects with content and metadata and add them to a Couchbase vector store using the `add_documents` function, associating them with unique IDs. Requires `uuid4` from `uuid` and `Document` from `langchain_core.documents`.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/vectorstores/couchbase.ipynb#_snippet_8

LANGUAGE: Python
CODE:
```
from uuid import uuid4

from langchain_core.documents import Document

document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)
```

----------------------------------------

TITLE: Executing Vectara Queries with Configuration (Python)
DESCRIPTION: Shows various examples of querying the Vectara index using the `vectara.vectara_query` method. Each example passes a different natural language query string and the previously configured `config` object, which includes the enabled `intelligent_query_rewriting`. Demonstrates how different types of queries (simple, filter-based, composite) are handled. Requires an initialized `Vectara` object and a `VectaraQueryConfig` object.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/retrievers/self_query/vectara_self_query.ipynb#_snippet_6

LANGUAGE: python
CODE:
```
# This example only specifies a relevant query\nvectara.vectara_query("What are movies about scientists", config)
```

LANGUAGE: python
CODE:
```
# This example only specifies a filter\nvectara.vectara_query("I want to watch a movie rated higher than 8.5", config)
```

LANGUAGE: python
CODE:
```
# This example specifies a query and a filter\nvectara.vectara_query("Has Greta Gerwig directed any movies about women", config)
```

LANGUAGE: python
CODE:
```
# This example specifies a composite filter\nvectara.vectara_query("What's a highly rated (above 8.5) science fiction film?", config)
```

LANGUAGE: python
CODE:
```
# This example specifies a query and composite filter\nvectara.vectara_query(\n    "What's a movie after 1990 but before 2005 that's all about toys, and preferably is animated",\n    config,\n)
```

----------------------------------------

TITLE: Update Chain with Example Messages (Python)
DESCRIPTION: Constructs a new LangChain chain by piping the input question through a prompt template that includes the `example_msgs` using `prompt.partial`. The output is then passed to a structured LLM, effectively incorporating the few-shot examples into the model's context for query analysis. Requires `langchain_core.prompts.MessagesPlaceholder` and `RunnablePassthrough`.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/how_to/query_few_shot.ipynb#_snippet_10

LANGUAGE: python
CODE:
```
from langchain_core.prompts import MessagesPlaceholder

query_analyzer_with_examples = (
    {"question": RunnablePassthrough()}
    | prompt.partial(examples=example_msgs)
    | structured_llm
)
```

----------------------------------------

TITLE: Initializing ChatOpenAI Model (Python)
DESCRIPTION: Imports and initializes a `ChatOpenAI` language model with a temperature of 0, typically used for deterministic responses.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/retrievers/arxiv.ipynb#_snippet_8

LANGUAGE: python
CODE:
```
# | output: false
# | echo: false

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
```

----------------------------------------

TITLE: Embed Single Text with TogetherEmbeddings (Python)
DESCRIPTION: Shows how to directly use the embed_query method of the TogetherEmbeddings object to generate an embedding vector for a single input string.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/text_embedding/together.ipynb#_snippet_5

LANGUAGE: python
CODE:
```
single_vector = embeddings.embed_query(text)
print(str(single_vector)[:100])  # Show the first 100 characters of the vector
```

----------------------------------------

TITLE: Indexing and Retrieving with Ollama Embeddings - LangChain - Python
DESCRIPTION: This snippet shows how to use the initialized `OllamaEmbeddings` object to create an `InMemoryVectorStore` from text and then use it as a retriever. It demonstrates indexing a sample text and retrieving the most similar document based on a query.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/text_embedding/ollama.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
# Create a vector store with a sample text
from langchain_core.vectorstores import InMemoryVectorStore

text = "LangChain is the framework for building context-aware reasoning applications"

vectorstore = InMemoryVectorStore.from_texts(
    [text],
    embedding=embeddings,
)

# Use the vectorstore as a retriever
retriever = vectorstore.as_retriever()

# Retrieve the most similar text
retrieved_documents = retriever.invoke("What is LangChain?")

# show the retrieved document's content
retrieved_documents[0].page_content
```

----------------------------------------

TITLE: Perform Semantic Search on Album Titles (SQL/Python)
DESCRIPTION: Demonstrates performing a semantic search on album titles by embedding a query string ("hope about the future") and using the `<->` operator to find the closest embeddings in the "Album" table.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/cookbook/retrieval_in_sql.ipynb#_snippet_18

LANGUAGE: python
CODE:
```
embeded_title = embeddings_model.embed_query("hope about the future")
query = (
    'SELECT "Album"."Title" FROM "Album" WHERE "Album"."embeddings" IS NOT NULL ORDER BY "embeddings" <-> '
    + f"'{embeded_title}' LIMIT 5"
)
db.run(query)
```
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def load_document():
    with open("documents/plunge.txt", "r", encoding="utf-8") as f:
        return f.read()

def setup_rag():
    # Load the document
    raw_text = load_document()
    
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    
    # Create embeddings and store them in Chroma
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_texts(texts, embeddings)
    
    # Create a retriever
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # Setup memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create the chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0),
        retriever=retriever,
        memory=memory,
        verbose=True
    )
    
    return qa

def main():
    # Load environment variables
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY in a .env file")
        return
    
    print("Setting up RAG system...")
    qa_chain = setup_rag()
    
    print("\nWelcome to the Plungė Information System!")
    print("You can ask questions about the city of Plungė. Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            break
            
        try:
            result = qa_chain({"question": question})
            print("\nAnswer:", result["answer"])
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

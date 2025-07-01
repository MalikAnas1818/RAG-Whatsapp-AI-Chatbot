from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env

def get_answer(directory = "faiss_index", question="What is NLP"):

    instruction_embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.load_local(
        directory,  # ✅ full path to your index
        instruction_embedding,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})



    llm = ChatGroq(model="llama3-70b-8192")
    
    prompt=PromptTemplate(
        template="""
        You are an AI assistant trained specifically in the fields of Artificial Intelligence, including Machine Learning (ML), Deep Learning (DL), Natural Language Processing (NLP), and Python programming.
        You must answer only questions that are directly related to these topics.
        If the user asks something unrelated (e.g., about sports, politics, general knowledge), politely decline and respond:
        I'm sorry, I am only trained in the AI domain (ML, DL, NLP, Python). I can't answer questions outside this field.

        {context}
        Question: {question}
        """,
        input_variables=['context','question']
    )

    

    chain = LLMChain(
    llm=llm,
    prompt=prompt
    )

    retriever_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retriever_docs)


    response = chain.invoke({
        "context": context_text,
        "question": question
        })
    
    return response["text"]


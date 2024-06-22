from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

llm = Ollama(model="llama3")

question_user = input("Hola, ¿cuál es tu pregunta?")

#respuesta = llm.invoke("Hola ¿quien eres?")

#print(respuesta)

loader = PyMuPDFLoader("src/navarro_ciberseguridad_datos_2023.pdf")
data_pdf = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)

docs = text_splitter.split_documents(data_pdf)

embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#print(docs[40])

v5 = Chroma.from_documents(
    documents=docs,
    embedding=embed_model,
    persist_directory="chroma_db_dir",
    collection_name="ia_ciberseguridad_data"
)

vectorstore = Chroma(
    embedding_function= embed_model,
    persist_directory="chroma_db_dir",
    collection_name="ia_ciberseguridad_data"
)
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})


custom_prompt_template = """Usa la siguiente información para responder a la pregunta del usuario. Si no sabes la respuesta, simplemente di que no lo sabes, no intentes inventar la respuesta

Contexto: {context}
Pregunta: {question}

Solo devuelve la respuesta útil a continuación y nada más y responde siempre en español

Respuesta útil:

"""

prompt = PromptTemplate(template=custom_prompt_template,
                        input_variables=['context', 'question'])

qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=retriever,
                                 return_source_documents=True,
                                 chain_type_kwargs={"prompt":prompt})

#response = qa.invoke({"query":"Qué son los sistemas agiles y cognitivos"})


response = qa.invoke({"query": question_user})


#response = qa.invoke({"query": "¿Qué es Adversarial machine learning con enfoque de black-box?"})


print(response['result'])



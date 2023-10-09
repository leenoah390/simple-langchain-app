import os
import sys
from langchain.document_loaders import WebBaseLoader, PDFMinerLoader
from langchain.document_loaders import OnlinePDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

OPENAI_API_KEY= 

# Defining a Variable to store the file path or article link you want to do question answering on.

webpage_path = input("Put the path to your webpage pdf: ")# example: 'https://arxiv.org/pdf/2211.12588.pdf'

# Defining a Loader object
# The loader object is specifc to the type of file you selected in the previous page.
# For webpage there is WebBaseLoader, for online pdfs (arxiv articles) there is OnlinePDFLoader

# loader = WebBaseLoader(webpage_path)
loader = OnlinePDFLoader(webpage_path)

# loader.load() starts to extract text/information from the provided file link.
# So if you are doing question answering on an Article.
# loader.load() step will extract all the text from that article link.
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)

# lets see what the 33rd piece of splitted text looks like
#print(all_splits[33].page_content)

# creating a vectorestore to put the vectorized text into
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))


# Our question. Try changing this and see what happens
question = "Can you compare the results of the GPT-3 backend to the Codex backend according to the paper?"

# document retrieval using ChromaDb and provided question
docs = vectorstore.similarity_search(question)

# lets see which text pieces are retrieved
#print(docs)

# defining which LLM to use.
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

# Defining the prompt
template = """Use the following pieces of a research paper to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# combining all the previous steps and creating a nice and clean Chain Object.

qa_chain = RetrievalQA.from_chain_type(
    llm, # llm we created in step 2A
    retriever=vectorstore.as_retriever(), # vector store we created in step 1C
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} # Prompt template we created in step 2B
)

# our question
question = input("enter a prompt: ")

# getting the answer
answer = qa_chain.run(question)
print(answer)
import requests
from pathlib import Path

from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import ConversationChain

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings


class ChatbotBackend:
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.db = None
        self.retriever = None
        self.sources = []
        self.fnames = []
        self.chain = None
        self.similarity_k = 1
        self.markdown_sources = []

    def reset_llm(self):
        self.llm = None
        self.embeddings = None
        self.chain = None

    def authenticate(self, api_key):
        def is_valid_openai_key(api_key):
            # doing this without using openai.api_key, as it propagates globally to all users
            headers = {"Authorization": f"Bearer {api_key}"}
            url = "https://api.openai.com/v1/engines"

            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                return True
            else:
                return False

        if is_valid_openai_key(api_key):
            # print("API key is valid.")
            # history = history + [[None, "API key set successfully!"]]
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key
            )
            self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            self.chain = ConversationChain(llm=self.llm, verbose=False)
        else:
            # print("Something went wrong, check your API key.")
            self.reset_llm()

    def generate_response(self, user_input, chat_history=None):
        # generate a response from the model
        # if the chatbot chain is a ConversationChain, use .predict
        # if the chatbot chain is a ConversationalRetrievalChain, use chatbot_chain({"question": user_input, "chat_history": []})
        if isinstance(self.chain, ConversationChain):
            return self.chain.predict(input=user_input)
        elif isinstance(self.chain, ConversationalRetrievalChain):
            return self.chain({"question": user_input, "chat_history": []})["answer"]
        else:
            return "Please paste your OpenAI key..."

    def update_chain(self):
        # if sources are added, switch chatbot chain to ConversationalRetrievalChain
        self.similarity_k = len(self.sources)
        retriever = self.db.as_retriever(search_kwargs={"k": self.similarity_k})
        if not isinstance(self.chain, ConversationalRetrievalChain):
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
            )
        else:
            self.chain.retriever = retriever

    def update_sources(self, fname):
        # add data to existing database for retrieval
        # call swap_chain when the first source is added
        # add data to existing database for retrieval
        if Path(fname).suffix==".txt":
            loader = TextLoader(file_path=f"{fname}")
            data = loader.load()
            self.fnames.append(f"{Path(fname).stem}{Path(fname).suffix}")
            self.markdown_sources.append(f"## {Path(fname).stem}{Path(fname).suffix}")
            self.sources.append(data[0].page_content)
            self.markdown_sources.append(data[0].page_content)
        else:
            loader = CSVLoader(file_path=f"{fname}")
            data = loader.load()
            self.fnames.append(f"{Path(fname).stem}{Path(fname).suffix}")
            self.markdown_sources.append(f"## {Path(fname).stem}{Path(fname).suffix}")
            for i in data:
                self.sources.append(i.page_content)
                self.markdown_sources.append(i.page_content)

        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        texts = text_splitter.split_documents(data)

        if not self.db:
            self.db = Chroma.from_documents(texts, self.embeddings)
        else:
            self.db.add_documents(texts)

        self.update_chain()

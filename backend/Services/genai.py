from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI
from langchain.chains.summarize import load_summarize_chain

# from google.oauth2.service_account import Credentials
# import sys, os
# sys.path.append(os.path.abspath('../'))

class YoutubeProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=0
        )

    def retrieve_youtube_documents(self, youtube_link: str, verbose = False):
        loader = YoutubeLoader.from_youtube_url(youtube_link, add_video_info = True)
        docs = loader.load()
        result = self.text_splitter.split_documents(docs)

        author = result[0].metadata['author']
        length = result[0].metadata['length']
        title = result[0].metadata['title']
        total_size = len(result)

        if verbose:
            print(f"author: {author}\n length: {length}\n title: {title}\n total_size: {total_size}")

        return result
    

class GeminiProcessor:
    def __init__(self, model_name, project):
        # import sys, os
        # root_dir = os.path.abspath('../')
        # key_path = root_dir + '\\' + credential_file_name + '.json'
        # self.credentials = Credentials.from_service_account_file(key_path)
        self.model = VertexAI(model_name=model_name, project= project)

    def generate_document_summary(self, documents: list, **args):
        chain_type = 'map_reduce' if len(documents) > 10 else 'stuff'
        chain = load_summarize_chain(
            llm= self.model,
            chain_type = chain_type,
            **args
        )
        return chain.run(documents)
    

# if __name__=='__main__':
#     credential_file_name = 'gemini-dynamo-credentials'
#     model_name ="gemini-1.5-pro-001"
#     project_id = 'gemini-dynamo-423808'
#     genai_processor = GeminiProcessor(model_name = model_name, 
#                                        project = project_id,
#                                     #    credential_file_name = credential_file_name 
#                                        )
#     documents = "This is a text just to check my credentials are working fine and model is responding properly. I will generate proper results later, meanwhile just use this text and check the code."
#     genai_processor.generate_document_summary(documents=documents)  
#     result = gemini_processor.generate_document_summary(documents=documents)
#     print(result)
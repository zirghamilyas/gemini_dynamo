from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI
from vertexai.generative_models import GenerativeModel
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)


# from google.oauth2.service_account import Credentials
# import sys, os
# sys.path.append(os.path.abspath('../'))
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
    
    def count_total_tokens(self, docs:list):
        temp_model = GenerativeModel('gemini-1.0-pro')
        total = 0
        # logger.info('Counting total tokens...')
        print('Counting total tokens...')

        from tqdm import tqdm
        for doc in tqdm(docs):
            total += temp_model.count_tokens(doc.page_content).total_tokens
            return total

    
    def get_model(self):
        return self.model
    

class YoutubeProcessor:
    def __init__(self, genai_processor: GeminiProcessor):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=0
        )
        self.GeminiProcessor = genai_processor

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
    
    def find_key_concepts(self, documents : list, group_size: int = 2):
        # Iterate through all documents of group size N and find key concepts
        if group_size > len(documents):
            raise ValueError("Group size is larger than the number of documents")
        # finding number of docs in each group
        num_docs_per_group = len(documents) // group_size + (len(documents) % group_size > 0)

        # split the document in chunks of size num_docs_per_group
        groups = [documents[i:i+num_docs_per_group] for i in range (0,len(documents), num_docs_per_group)]

        batch_concepts = []

        # logger.info('Finding key concepts...')
        print('Finding key concepts...')
        for group in tqdm(groups):
            # combine content of documents per group
            group_content = ""

            for doc in group:
                group_content += doc.page_content

            # prompt for finding concepts
            prompt = PromptTemplate(
                template = """
                Find and define key concepts or terms found in the text:
                {text}

                Respond in the following format as a string separating each concept with a comma:
                "concept": "definition"
                """,
                input_variable = ['text']
            )

            # create chain
            chain = prompt | self.GeminiProcessor.model

            # Run chain
            concept = chain.invoke({'text': group_content})
            batch_concepts.append(concept)
        
        return batch_concepts

    


    

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
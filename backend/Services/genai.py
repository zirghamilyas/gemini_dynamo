from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI
from vertexai.generative_models import GenerativeModel
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import json

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
        # logging.info('Counting total tokens...')
        print('Counting total billable characters...')

        from tqdm import tqdm
        for doc in tqdm(docs):
            total += temp_model.count_tokens(doc.page_content).total_billable_characters
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
        total_billable_characters = self.GeminiProcessor.count_total_tokens(result)

        if verbose:
            print(f"author: {author}\n length: {length}\n title: {title}\n total_size: {total_size}\n total_billable_characters: {total_billable_characters}")

        return result
    
    def find_key_concepts(self, documents : list, sample_size: int = 0, verbose = False):
        # Iterate through all documents of group size N and find key concepts
        if sample_size > len(documents):
            raise ValueError("Group size is larger than the number of documents")
        
        # optimized sample size for no input i.e. sample_size = 0
        if sample_size == 0:
            sample_size = len(documents) // 5
            if verbose:
                print(f"Sample size is not provided. Using sample size: {sample_size}, so to get 5 documents per group")
        # finding number of docs in each group
        num_docs_per_group = len(documents) // sample_size + (len(documents) % sample_size > 0)

        # check threshold for response quality
        if num_docs_per_group >= 10: 
           raise ValueError("Sample size is small and it will reduce the output quality significantly because the number of documents per group is more than 7. Please try a smaller sample_size")
        elif num_docs_per_group < 3:
            raise ValueError("Sample size is large and it will reduce the output quality significantly because the number of documents per group is less than 3. Please try a larger sample_size")
        
        print(f'Num of doc per group {num_docs_per_group}')

        # split the document in chunks of size num_docs_per_group
        groups = [documents[i:i+num_docs_per_group] for i in range (0,len(documents), num_docs_per_group)]

        batch_concepts = []
        total_batch_cost = 0

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

                Respond as a JSON object with the following format:
                {{"concept": "definition", "concept": "definition", ...}}
                Make sure to separate each concept and definition with a comma.
                """,
                input_variable = ['text']
            )

            # create chain
            chain = prompt | self.GeminiProcessor.model

            error = 0
            counter = 0
            while error == 0:
                counter += 1
                # Run chain
                output_concept = chain.invoke({'text': group_content})
                
                try:
                    # Convert the JSON String to a dictionary
                    processed_concepts = json.loads(output_concept)
                    error = 1
                except json.JSONDecodeError:
                    print("Failed to decode output_concept into JSON.")
                    continue  # Skip this iteration if JSON decoding fails
            
            dict = {}                             
            for key, value in processed_concepts.items():
                dict = {'term': key, 'definition': value}
                batch_concepts.append(dict)

            # Post processing Observation
            if verbose:
                total_input_char = len(group_content)
                total_input_cost = (total_input_char/1000) * 0.000125 * counter

                print(f' Running chain on {len(group)} documents')
                print(f'Total input characters: {total_input_char}')
                print(f'Total input cost: {total_input_cost}')

                # Output cost
                total_output_char = len(output_concept)
                total_output_cost = (total_output_char/1000) * 0.000375 * counter

                print(f'Total output characters: {total_output_char}')
                print(f'Total output cost: {total_output_cost}')

                # Current batch cost
                batch_cost = total_input_cost + total_output_cost
                print(f'Current batch cost: {batch_cost}')
            else:
                total_input_char = len(group_content)
                total_input_cost = (total_input_char/1000) * 0.000125 * counter
                total_output_char = len(output_concept)
                total_output_cost = (total_output_char/1000) * 0.000375 * counter

                batch_cost = total_input_cost + total_output_cost
            
            total_batch_cost += batch_cost

        # Converting JSON strings into pthon dictionary
        # processed_concepts = [concept for concept in batch_concepts]

        
        # Total analysis cost
        print(f'Total analysis cost in dollars: ${total_batch_cost}')
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
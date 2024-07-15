from openai import AzureOpenAI,OpenAI

AZURE_OPENAI_KEY= ""#r"9ac3678dcfc34abb86e936878987f3a9"
AZURE_DEPLOYMENT_NAME = r'gpt4o'
AZURE_API_VERSION = "2024-02-15-preview"
AZURE_OPENAI_ENDPOINT = r"https://oai0-yrr5qcssl7kgi.openai.azure.com/"

PROMPT_PDF_PARSER = open("data_ingestion/gpt_4_o/prompt_template/pdf_parser.txt","r").read()

# CLIENT = AzureOpenAI(
#     api_key = AZURE_OPENAI_KEY,
#     api_version = AZURE_API_VERSION,
#     azure_endpoint = AZURE_OPENAI_ENDPOINT
# )
OPENAI_KEY = ""
MODEL_NAME = "gpt-4o"
CLIENT =  OpenAI(
    api_key=OPENAI_KEY
)

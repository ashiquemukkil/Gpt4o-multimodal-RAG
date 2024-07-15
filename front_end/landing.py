
import os
from data_ingestion.gpt_4_o.config import OPENAI_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

from dataclasses import dataclass
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

from langchain.chains.summarize import load_summarize_chain
from time import monotonic

from pytube import YouTube
from pydub import AudioSegment
import  whisper
import tempfile

from kb import save_to_db,find_relevant_chunk

@dataclass
class Message:
    actor: str
    payload: str

def summarize_text(docs):
    llm = OpenAI()
    prompt_template = """Write a concise summary of the following  Youtube Video Transcript:

    Transcript:
    {text}

    Return summary as HTML format 
    Summary:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    verbose = True

    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt, combine_prompt=prompt, verbose=verbose)

    summary = chain.run(docs)
    return summary



def download_audio_from_youtube(video_url):
    yt = YouTube(video_url)
    stream = yt.streams.filter(only_audio=True).first()
    temp_dir = tempfile.mkdtemp()
    audio_file_path = os.path.join(temp_dir, 'audio.mp4')
    stream.download(filename=audio_file_path)
    return audio_file_path

# Function to convert audio to WAV format
def convert_to_wav(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    wav_file_path = audio_file_path.replace('.mp4', '.wav')
    audio.export(wav_file_path, format='wav')
    return wav_file_path

def get_whisper_model():
    return whisper.load_model("base")  # Load the Whisper model

# Function to transcribe audio using Whisper
def transcribe_audio(wav_file_path):
    model = st.session_state["whisper"] 
    result = model.transcribe(wav_file_path)
    return result['text']

# @st.cache_resource
def get_llm():
    return OpenAI(temperature=0.9) 


def get_llm_chain():
    template = """You are a nice chatbot having a conversation with a human.

    Previous conversation:
    {chat_history}

    Context:
    {context}

    New human question: {human_input}
    Response:"""
    prompt_template = PromptTemplate(template=template,input_variables = ["chat_history","human_input","context"])
    # Notice that we need to align the `memory_key`
    memory = ConversationBufferMemory(memory_key="chat_history",input_key="human_input")
    conversation = load_qa_chain(
        llm=get_llm(),
        prompt=prompt_template,
        memory=memory
    )
    return conversation


USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"


def initialize_session_state():
    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hi!How can I help you?")]
    if "llm_chain" not in st.session_state:
        st.session_state["llm_chain"] = get_llm_chain()
    if "whisper" not in st.session_state:
        st.session_state["whisper"] = get_whisper_model()



def get_llm_chain_from_session() -> ConversationChain:
    return st.session_state["llm_chain"]


initialize_session_state()

msg: Message
for msg in st.session_state[MESSAGES]:
    st.chat_message(msg.actor).write(msg.payload)


with st.sidebar:
    if "url" not in st.session_state:
        placeholder = st.empty()
        url = placeholder.text_input("past your youtube url")
        with st.spinner("Generating transcript"):
            if url:
                st.write("Downloading audio from YouTube video...")
                audio_file_path = download_audio_from_youtube(url)

                st.write("Converting audio to WAV format...")
                wav_file_path = convert_to_wav(audio_file_path)

                st.write("Transcribing audio...")
                transcript = transcribe_audio(wav_file_path)

                st.write("Transcript:")
                st.write(transcript)
        
                # Clean up temporary files
                os.remove(audio_file_path)
                os.remove(wav_file_path)

                st.write("Chunking :")
                db = save_to_db(transcript)
                st.session_state["db"] = db

                st.session_state["url"] = url

if "url" in st.session_state:
    prompt: str = st.chat_input("Enter a prompt here")
    
    if st.button("SUMMARIZE"):
        st.title("Summary")
        chunks = find_relevant_chunk("",st.session_state["db"],k=100)
        summary = summarize_text(chunks)
        st.write(summary)
else:
    st.chat_message(ASSISTANT).write("Youtube url required")
    prompt = None


if prompt:
    st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
    st.chat_message(USER).write(prompt)

    with st.spinner("Please wait.."):
        llm_chain = get_llm_chain_from_session()
        context = find_relevant_chunk(prompt,st.session_state["db"])
        response: str = llm_chain({"input_documents": context,"human_input":prompt},return_only_outputs=True)["output_text"]
        st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
        st.chat_message(ASSISTANT).write(response)
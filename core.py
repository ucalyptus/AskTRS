import whisper
import pytube
import faiss
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.vectorstores.base import VectorStoreRetriever
import os
import tempfile

SEGMENT_SIZE = 1000
CHUNK_SIZE = 1500

def transcribe_video(video_link, temp_dir):
    video = pytube.YouTube(video_link)
    audio = video.streams.get_audio_only()
    audio_file = audio.download(output_path=temp_dir)
    model = whisper.load_model("base")
    transcription = model.transcribe(audio_file)
    return transcription['text']

def create_vector_store(text, temp_dir, api_key):
    def store_segments(text):
        segments = [{'text': text[i:i+SEGMENT_SIZE], 'start': i} for i in range(0, len(text), SEGMENT_SIZE)]

        texts = []
        start_indices = []

        for segment in segments:
            texts.append(segment['text'])
            start_indices.append(f"Character {segment['start']}")

        return texts, start_indices

    texts, start_indices = store_segments(text)

    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, separator="\n")
    docs = []
    metadatas = []
    for i, d in enumerate(texts):
        splits = text_splitter.split_text(d)
        docs.extend(splits)
        metadatas.extend([{"source": start_indices[i]}] * len(splits))

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    store = FAISS.from_texts(docs, embeddings, metadatas=metadatas)
    index_path = os.path.join(temp_dir, "docs.index")
    faiss.write_index(store.index, index_path)
    return store

def query_chain(store, question, api_key):
    retri = VectorStoreRetriever(vectorstore=store)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0, openai_api_key=api_key), retriever=retri)
    result = chain({"question": question})
    return result['answer'], result['sources']

def get_answer(api_key, video_link, question):
    with tempfile.TemporaryDirectory() as temp_dir:
        transcription = transcribe_video(video_link, temp_dir)
        store = create_vector_store(transcription, temp_dir, api_key)
        answer, sources = query_chain(store, question, api_key)
    return answer, sources

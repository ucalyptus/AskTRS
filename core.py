import whisper
import pytube
import faiss
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.vectorstores.base import VectorStoreRetriever
import os
import tempfile

def transcribe_video(video_link, temp_dir):
    video = pytube.YouTube(video_link)
    audio = video.streams.get_audio_only()
    audio_file = audio.download(output_path=temp_dir)
    model = whisper.load_model("base")
    transcription = model.transcribe(audio_file)
    return transcription['text']

def create_vector_store(text, temp_dir):
    def store_segments(text):
        segment_size = 1000
        segments = [{'text': text[i:i+segment_size], 'start': i} for i in range(0, len(text), segment_size)]

        texts = []
        start_times = []

        for segment in segments:
            text = segment['text']
            start = segment['start']

            start_datetime = datetime.fromtimestamp(start)
            formatted_start_time = start_datetime.strftime('%H:%M:%S')

            texts.append(text)
            start_times.append(formatted_start_time)

        return texts, start_times

    texts, start_times = store_segments(text)

    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    docs = []
    metadatas = []
    for i, d in enumerate(texts):
        splits = text_splitter.split_text(d)
        docs.extend(splits)
        metadatas.extend([{"source": start_times[i]}] * len(splits))

    embeddings = OpenAIEmbeddings()
    store = FAISS.from_texts(docs, embeddings, metadatas=metadatas)
    index_path = os.path.join(temp_dir, "docs.index")
    faiss.write_index(store.index, index_path)
    return store

def query_chain(store, question):
    retri = VectorStoreRetriever(vectorstore=store)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), retriever=retri)
    result = chain({"question": question})
    return result['answer'], result['sources']

def get_answer(api_key, video_link, question):
    os.environ["OPENAI_API_KEY"] = api_key
    with tempfile.TemporaryDirectory() as temp_dir:
        transcription = transcribe_video(video_link, temp_dir)
        store = create_vector_store(transcription, temp_dir)
        answer, sources = query_chain(store, question)
    return answer, sources

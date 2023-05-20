import whisper 
import pytube
import gradio as gr
import openai
import faiss
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.vectorstores.base import VectorStoreRetriever
import os

def get_answer(api_key, video_link, question):
    os.environ["OPENAI_API_KEY"] = api_key

    video = pytube.YouTube(video_link)
    audio = video.streams.get_audio_only()
    fn = audio.download(output_path="tmp.mp3")
    model = whisper.load_model("base")
    transcription = model.transcribe(fn)
    res = transcription['text']

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

    texts, start_times = store_segments(res)

    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    docs = []
    metadatas = []
    for i, d in enumerate(texts):
        splits = text_splitter.split_text(d)
        docs.extend(splits)
        metadatas.extend([{"source": start_times[i]}] * len(splits))

    embeddings = OpenAIEmbeddings()
    store = FAISS.from_texts(docs, embeddings, metadatas=metadatas)
    faiss.write_index(store.index, "docs.index")

    retri = VectorStoreRetriever(vectorstore=store)

    chain = RetrievalQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), retriever=retri)

    result = chain({"question": question})

    return result['answer'], result['sources']

iface = gr.Interface(
    fn=get_answer,
    inputs=["text", "text", "text"],
    outputs=["text", "text"],
)

iface.queue().launch()

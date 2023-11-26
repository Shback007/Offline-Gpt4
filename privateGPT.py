#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import chromadb
import os
import argparse
import time
from flask import Flask, request, render_template_string

app = Flask(__name__)

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

from constants import CHROMA_SETTINGS

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        if query.strip() == "":
            return render_template_string("Please enter a query.")
        
        # Process the query and get the answer
        answer, docs = process_query(query)

        return render_template_string(
            """
            <body style="font-family: Times New Roman, sans-serif; background-color: #f0f0f0; margin: 0; padding: 0;">

                <div style="background-color: #007bff; text-align: center; padding: 20px;"></div>

                <div style="margin: 20px; padding: 20px; background-color: #ffffff; border-radius: 8px;">

                    <form method="post" style="max-width: 800px; margin: 0 auto;">
                        <label for="query" style="font-size: 18px; display: block; margin-bottom: 0px; font-size: 2rem;"><b>Enter a query:</b></label>
                        <br>
            
                        <input type="text" id="query" name="query" style="height: 5rem; width: 100%; padding: 10px; font-size: 16px; solid #ccc; border-radius: 5px; margin-bottom: 10px;" required>
                        <br>
                        
                        <input type="submit" value="Submit" style="background-color: #007bff; color: white; border: none; padding: 10px 20px; font-size: 16px; border-radius: 4px; cursor: pointer;">
                    </form>

                </div>

            </body>

            <h2 style="font-size: 18px; display: block; margin-bottom: 0px; font-size: 2rem; margin-left: 3%">Answer:</h2>
            <div style="background-color: #f0f0f0; border: 1px solid #ddd; border-radius: 5px; padding: 20px; margin: 20px;">
                <p style="font-size: 1.3rem; color: #333; text-align: justify; line-height: 1.4;">{{ answer }}</p>
            </div>
            
            """,
            answer=answer,
            docs=docs
        )

    return render_template_string(
        """
        <body style="font-family: Times New Roman, sans-serif; background-color: #f0f0f0; margin: 0; padding: 0;">

                <div style="background-color: #007bff; text-align: center; padding: 20px;"></div>

                <div style="margin: 20px; padding: 20px; background-color: #ffffff; border-radius: 8px;">

                    <form method="post" style="max-width: 800px; margin: 0 auto;">
                        <label for="query" style="font-size: 18px; display: block; margin-bottom: 0px; font-size: 2rem;"><b>Enter a query:</b></label>
                        <br>
            
                        <input type="text" id="query" name="query" style="height: 5rem; width: 100%; padding: 10px; font-size: 16px; solid #ccc; border-radius: 5px; margin-bottom: 10px;" required>
                        <br>
                        
                        <input type="submit" value="Submit" style="background-color: #007bff; color: white; border: none; padding: 10px 20px; font-size: 16px; border-radius: 4px; cursor: pointer;">
                    </form>

                </div>

        </body>
        """
    )

def process_query(query):
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS, path=persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=[], verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=[], verbose=False)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    
    # Get the answer from the chain
    start = time.time()
    res = qa(query)
    answer, docs = res['result'], res['source_documents']
    end = time.time()

    return answer, docs

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

if __name__ == "__main__":
    app.run(debug=True)

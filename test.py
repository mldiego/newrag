import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# Set the Ollama endpoint (if it's not the default)
# os.environ["OLLAMA_HOST"] = "http://localhost:11434"

# 1. Load the PDF file
# Replace 'your_document.pdf' with your file path
try:
    loader = PyPDFLoader("data/LNCS.pdf")
    docs = loader.load()
    print(docs)
except FileNotFoundError:
    print("Error: The file 'your_document.pdf' was not found. Please ensure the file exists.")
    exit()

# 2. Initialize the Ollama embedding model
# Ensure 'mxbai-embed-large' is pulled via 'ollama pull mxbai-embed-large'
try:
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
except Exception as e:
    print(f"Error initializing OllamaEmbeddings: {e}")
    print("Please check that Ollama is running and the specified model is pulled.")
    exit()

# 3. Perform semantic chunking
# The SemanticChunker uses a character-based splitter and then merges chunks
# based on the semantic similarity calculated by the embeddings.
semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)

# 4. Split the document into semantic chunks
semantic_chunks = semantic_chunker.split_documents(docs)

# 5. Print some information to verify the process
print(f"Total number of chunks created: {len(semantic_chunks)}")
print("\n--- First Chunk Content ---")
print(semantic_chunks[0].page_content)

# You can now use these semantic_chunks for your RAG system,
# e.g., storing them in a vector database like ChromaDB or Milvus.

# https://github.com/jina-ai/late-chunking


# Do some late chunking now
from transformers import AutoModel
from transformers import AutoTokenizer
from chunked_pooling import chunked_pooling, chunk_by_sentences

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)


# Chunk by sentences
input_text = "Berlin is the capital and largest city of Germany, both by area and by population. Its more than 3.85 million inhabitants make it the European Union's most populous city, as measured by population within city limits. The city is also one of the states of Germany, and is the third smallest state in the country in terms of area."
# determine chunks
chunks, span_annotations = chunk_by_sentences(input_text, tokenizer)
print('Chunks:\n- "' + '"\n- "'.join(chunks) + '"')


# Now we encode the chunks with the traditional and the context-sensitive chunked pooling method
# chunk before
embeddings_traditional_chunking = model.encode(chunks)
# chunk afterwards (context-sensitive chunked pooling)
inputs = tokenizer(input_text, return_tensors='pt')
model_output = model(**inputs)
embeddings = chunked_pooling(model_output, [span_annotations])[0]


# Now, compare similarity with chunks
import numpy as np

cos_sim = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

berlin_embedding = model.encode('Berlin')

for chunk, new_embedding, trad_embeddings in zip(chunks, embeddings, embeddings_traditional_chunking):
    print(f'similarity_new("Berlin", "{chunk}"):', cos_sim(berlin_embedding, new_embedding))
    print(f'similarity_trad("Berlin", "{chunk}"):', cos_sim(berlin_embedding, trad_embeddings))
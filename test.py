import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# For late chunking
from transformers import AutoModel
from transformers import AutoTokenizer
from chunked_pooling import chunked_pooling, chunk_by_sentences

# For similarity search
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random
import numpy as np


def chunking(chunk_type,text):
    # Initialize 
    tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    inputs = tokenizer(text, return_tensors='pt')
    model_output = model(**inputs)
    # Choose chunking + embeddings
    if chunk_type=='semantic':
        embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
        chunker = SemanticChunker(
            embedding_model,
            breakpoint_threshold_type="percentile"
        )
        chunks,spans = split_with_spans(text, chunker)
        embeddings = chunked_pooling(model_output, [spans])[0]
    elif chunk_type=='late':
        chunks, spans = chunk_by_sentences(text, tokenizer)
        embeddings = chunked_pooling(model_output, [spans])[0]
    elif chunk_type=='traditional':
        chunks, spans = chunk_by_sentences(text, tokenizer)
        encodings = model.encode(chunks)
        embeddings = []
        for j in range(encodings.shape[0]):
            embeddings.append(encodings[j])
        # embeddings = chunks
    # elif chunk_type=='hierarchical':
    #     print("To be implemented")
    else:
        return "Chunker not implemented yet"
    return chunks, embeddings
    

# FOR STORING EMBEDDINGS AND SEARCH
class MockVectorStore:
    def __init__(self, docs, embedding_model, chunk_type):
        self.docs = docs
        self.chunks = []
        self.embeddings = []
        self.model = embedding_model
        # chunk afterwards (context-sensitive chunked pooling) - latent
        for text in docs:
            chunks, embeddings = chunking(chunk_type, text)
            for i in chunks:
                self.chunks.append(i)
            for emb in embeddings:
                self.embeddings.append(emb)

    def semantic_search(self, query_text, k=10):
        # A mock semantic search using cosine similarity
        query_embedding = self.model.encode(query_text)
        similarities = []
        for i in self.embeddings:
            similarities.append(cos_sim(query_embedding, i))

        sorted_indices = np.argsort(similarities)[::-1]
        top_k_indices = sorted_indices[:k]
        
        results = []
        for k in top_k_indices:
            results.append(self.chunks[k])
            
        return results


    def threshold_search(self, query_text, threshold = 0.7):
        # A mock semantic search using cosine similarity
        query_embedding = self.model.encode(query_text)
        similarities = []
        for i in self.embeddings:
            similarities.append(cos_sim(query_embedding, i))

        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for k in sorted_indices:
            if similarities[k] >= threshold:
                results.append(self.chunks[k])
            else:
                break
            
        return results


cos_sim = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# Helper functions
# For comparison, let's create the span annotations for the langchain one as well
def split_with_spans(text, chunker):
    start_index = 0
    documents_with_spans = []
    chunks = []
    spans = []

    # Use the splitter to get the content of the chunks
    split_contents = chunker.split_text(text)

    for chunk_content in split_contents:
        # Find the starting position of the chunk in the original text
        # start_char = text.find(chunk_content, start_index)
        # if start_char == -1:
        #     # Fallback for overlaps or slight differences
        #     start_char = start_index

        end_index = start_index + len(chunk_content)

        chunks.append(chunk_content)
        spans.append((start_index, end_index))

        # Update the start_index for the next search
        start_index = end_index + 1

    return chunks,spans


# save results
def save_output_file(chunk_type, query, arg1, results, answer = None, default_filename="output.txt"):
    base_name, extension = os.path.splitext(default_filename)
    filename = default_filename
    counter = 1

    while os.path.exists(filename):
        filename = f"{base_name}_{counter}{extension}"
        counter += 1

    data = "Chunking Type: "+ chunk_type + "\n" + "User query/keyword: " + query + "\n" + "Threshold/topK: " + str(arg1) + "\n"

    with open(filename, 'w') as f:
        f.write(data)
        f.write("\n\n")
        f.write("Chunks: \n")
        for r in results:
            f.write(" - " + r+"\n")
        if answer:
            f.write("\n\nANSWER\n\n")
            f.write(answer)
    print(f"Data saved to: {filename}")
    return 



#############################################
##############  Ollama chat  ################
#############################################

def ollama_rag_request(model_name: str, context: str, user_query: str):
    """
    Sends a request to a local Ollama model with context and a user query
    and prints the model's response.
    
    Args:
        model_name: The name of the model to use (e.g., 'llama3').
        context: The specific context/document retrieved from the RAG system.
        user_query: The question from the user.
    """
    # 1. Define the RAG prompt structure
    # The system role is used to provide instructions and the context.
    SEPARATOR = "\n\n--- DOCUMENT CHUNK ---\n\n"
    full_context = SEPARATOR.join(context)
    system_message = (
        f"You are an expert Q&A assistant. Use ONLY the following context to answer the user's question. "
        f"If the answer cannot be found in the context, state that explicitly. "
        f"Context: {full_context}"
    )

    # 2. Prepare the messages list for the ollama.chat endpoint
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query}
    ]

    print(f"--- Sending request to model: {model_name} ---")

    # 3. Call the Ollama API
    try:
        response = ollama.chat(
            model=model_name,
            messages=messages
        )

        # 4. Print the final response content
        # print("\n--- Model Response ---")
        # print(response['message']['content'])
        # print("----------------------")
        
    except Exception as e:
        print(f"\nError connecting to Ollama: {e}")
        print("Please ensure the Ollama server is running and the specified model is pulled.")
        print("e.g., Run 'ollama pull llama3' in your terminal.")

    return response['message']['content']


#######################################
###########     DATA      #############
#######################################


# Data to include in db
berlin = "Berlin is the capital and largest city of Germany, both by area and by population. Its more than 3.85 million inhabitants make it the European Union's most populous city, as measured by population within city limits. The city is also one of the states of Germany, and is the third smallest state in the country in terms of area."
mj = """
Michael Jeffrey Jordan (born February 17, 1963), also known by his initials MJ,[8] is an American businessman, former professional basketball and baseball player, who is a minority owner of the Charlotte Hornets of the National Basketball Association (NBA). He played 15 seasons in the NBA between 1984-2003, winning six NBA championships with the Chicago Bulls. Widely considered to be one of the greatest players of all time,[9][10][11] he was integral in popularizing basketball and the NBA around the world in the 1980s and 1990s,[12] becoming a global cultural icon.[13]

Jordan played college basketball with the North Carolina Tar Heels. As a freshman, he was a member of the Tar Heels' national championship team in 1982.[5] Jordan joined the Bulls in 1984 as the third overall draft pick[5][14] and emerged as a league star, entertaining crowds with his prolific scoring while gaining a reputation as one of the best defensive players.[15] His leaping ability, demonstrated by performing slam dunks from the free-throw line in Slam Dunk Contests, earned him the nicknames "Air Jordan" and "His Airness".[5] Jordan won his first NBA title with the Bulls in 1991 and followed that with titles in 1992 and 1993, securing a three-peat. Citing physical and mental exhaustion from basketball and superstardom, Jordan abruptly retired before the 1993–94 NBA season to play Minor League Baseball in the Chicago White Sox organization. He returned to the Bulls in 1995 and led them to three more championships in 1996, 1997, and 1998, as well as a then-record 72 regular season wins in the 1995–96 NBA season.[5] Jordan retired for the second time in 1999, returning for two NBA seasons from 2001 to 2003 as a member of the Washington Wizards.[5][14] He was selected to play for the United States national team during his college and NBA careers, winning four gold medals—at the 1983 Pan American Games, 1984 Summer Olympics, 1992 Tournament of the Americas and 1992 Summer Olympics—while also being undefeated.[16]

Jordan's individual accolades include six NBA Finals Most Valuable Player (MVP) awards, ten NBA scoring titles (both all-time records), five NBA MVP awards, 10 All-NBA First Team designations, nine All-Defensive First Team honors, fourteen NBA All-Star Game selections, three NBA All-Star Game MVP awards, and three NBA steals titles.[14] He holds the NBA records for career regular season scoring average (30.1 points per game) and career playoff scoring average (33.4 points per game).[17] He is one of only eight players to achieve the basketball Triple Crown. In 1999, Jordan was named the 20th century's greatest North American athlete by ESPN and was second to Babe Ruth on the Associated Press' list of athletes of the century.[5] Jordan was twice inducted into the Naismith Memorial Basketball Hall of Fame, once in 2009 for his individual career,[18] and in 2010 as part of the 1992 United States men's Olympic basketball team ("The Dream Team").[19] He became a member of the United States Olympic Hall of Fame in 2009,[20] an individual member of the FIBA Hall of Fame in 2015 and a "Dream Team" member in 2017.[21][22] Jordan was named to the NBA 75th Anniversary Team in 2021.[23] The trophy for the NBA Most Valuable Player Award is named in his honor.

One of the most effectively marketed athletes ever, Jordan made many product endorsements.[12][24] He fueled the success of Nike's Air Jordan sneakers, which were introduced in 1984 and remain popular.[25] Jordan starred as himself in the live-action/animation hybrid film Space Jam (1996) and was the focus of the Emmy-winning documentary series The Last Dance (2020). He became part-owner and head of basketball operations for the Charlotte Hornets (then named the Bobcats) in 2006 and bought a controlling interest in 2010, before selling his majority stake in 2023. Jordan is a co-owner of 23XI Racing in the NASCAR Cup Series. In 2014, he became the first billionaire player in NBA history.[26] In 2016, President Barack Obama awarded Jordan the Presidential Medal of Freedom.[27] As of 2025, his net worth is estimated at $3.8 billion by Forbes,[28] making him one of the richest celebrities.
"""
cs_text = """
Computer science is the study of computation, information, and automation. Computer science spans theoretical disciplines (such as algorithms, theory of computation, and information theory) to applied disciplines (including the design and implementation of hardware and software). Algorithms and data structures are central to computer science. The theory of computation concerns abstract models of computation and general classes of problems that can be solved using them. The fields of cryptography and computer security involve studying the means for secure communication and preventing security vulnerabilities. Computer graphics and computational geometry address the generation of images. Programming language theory considers different ways to describe computational processes, and database theory concerns the management of repositories of data. Human–computer interaction investigates the interfaces through which humans and computers interact, and software engineering focuses on the design and principles behind developing software. Areas such as operating systems, networks and embedded systems investigate the principles and design behind complex systems. Computer architecture describes the construction of computer components and computer-operated equipment. Artificial intelligence and machine learning aim to synthesize goal-orientated processes such as problem-solving, decision-making, environmental adaptation, planning and learning found in humans and animals. Within artificial intelligence, computer vision aims to understand and process image and video data, while natural language processing aims to understand and process textual and linguistic data. The fundamental concern of computer science is determining what can and cannot be automated.The Turing Award is generally recognized as the highest distinction in computer science.
"""
s065 = """
Generated S065-Style Standard Document Text
TITLE: S065-B/SRD — Automated System Interface and Data Validation Standard, Revision 2.1

1.0 INTRODUCTION AND SCOPE

This document, S065-B/SRD, defines the mandatory technical specifications and operational protocols for the secure, automated transfer of clinical trial data between sponsor systems and the central regulatory body's database. The scope of this standard encompasses all data streams related to Protocol 734-X, specifically focusing on patient-reported outcomes (PROs), adverse event reporting (AERs), and investigational product accountability (IPA) logs. All data exchanges must be formatted in accordance with the HL7 FHIR standard, version R4, with specific profiles as detailed in Section 3.2. Data integrity is paramount, and non-compliance with these specifications will result in an automated rejection of the data submission.

2.0 DATA TRANSFER PROTOCOLS

2.1 Security and Authentication

All connections must be established via a TLS 1.3 encrypted channel. Each data stream must be accompanied by a digital signature using an SHA-256 hash and a Public Key Infrastructure (PKI) certificate issued by the approved Certificate Authority (CA) as designated in Appendix A. Client-side authentication requires a valid client certificate and a rotating API key, which must be refreshed every 24 hours. Failure to authenticate correctly will terminate the connection and log a security event in the central audit trail.

2.2 File Transmission and Archiving

Data payloads shall be transmitted as a compressed JSON file, with a maximum size of 50 MB per transmission. Submissions exceeding this limit must be partitioned into multiple files. Upon successful receipt, the system will generate a unique transaction ID and archive the file for a period of no less than seven (7) years. The sponsor is responsible for retaining a local copy of all submitted data and the corresponding transaction IDs.

3.0 DATA CONTENT AND VALIDATION

3.1 Patient-Reported Outcomes (PROs)

PRO data must adhere to the CDISC SDTM standard, version 1.6. Each record must include a unique patient identifier, the date of response, and the specific instrument used (e.g., EQ-5D-5L, QLQ-C30). A logical check will be performed to ensure that all response values fall within the permissible range for the specified instrument. Any out-of-range values will be flagged, and the entire batch will be returned for correction with a detailed error report.

3.2 Adverse Event Reporting (AERs)

AER data submissions must use the MedDRA coding system for all adverse events, up to the lowest level of term (LLT). Each report must include the event onset date, severity, and a causality assessment in relation to the investigational product. The system will cross-reference all submitted AERs against a predefined list of high-priority events and will automatically trigger a critical alert for immediate manual review if a match is found.

3.3 Investigational Product Accountability (IPA) Logs

IPA logs are required to be submitted on a monthly basis. The data must include the batch number, expiration date, and a detailed record of the product dispensed, returned, or destroyed. All quantities must be reconciled with the initial shipment manifest. Any discrepancy greater than a 2% variance must be justified in a separate discrepancy report submitted alongside the data. Failure to provide a valid justification will result in an official compliance notice.

4.0 GOVERNANCE AND COMPLIANCE

This standard is subject to periodic review and revision. Any changes will be communicated via official channels and will be published with a 90-day grace period before mandatory enforcement. Non-compliance with this standard, including but not limited to repeated failed submissions or data integrity issues, may lead to a formal investigation and potential suspension of the sponsor's data submission privileges. This document supersedes and replaces all previous versions of S065-A/SRD.
"""

documents = [
    berlin,
    cs_text,
    mj,
    s065,
]

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)

# define chunk types
chunk_types = ["late","semantic","traditional"]
# define user queries
user_query = ["Where did Michael Jordan play basketball at?", "Describe data content and validation for s065", "What are the main branches of computer science?", "Where is Paris?", "Who is Berlin?", "What is the capital of Germany?"]
# define search keywords
keywords = ["Michael Jordan", "basketball", "data content", "validation", "computer science", "branches", "Paris", "capital", "Germany"]
# define number of chunks to retrieve
topK = [3,4,5]
# threshold value to search
threshold = [0.8, 0.75, 0.7, 0.65]


# llm for answering based on chunks
llm = "qwen3:4b-instruct-2507-q4_K_M"

import ollama



for ct in chunk_types:
    # Create vector store
    print("Chunk type : "+ ct)
    vector_store = MockVectorStore(documents, model, ct)
    for uq in user_query:
        print("User query : "+ uq)
        for th in threshold:
            print("Threshold : "+str(th))
            # threshold search
            results = vector_store.threshold_search(uq, th)
            answer = ollama_rag_request(llm,results,uq)
            save_output_file(ct, uq, th, results, answer, default_filename="output/"+ct+"/threshold_search_query_"+str(th)+".txt")
        for k in topK:
            print("TopK : "+str(k))
            # top_k search
            results = vector_store.semantic_search(uq, k)
            answer = ollama_rag_request(llm,results,uq)
            save_output_file(ct, uq, k, results, answer, default_filename="output/"+ct+"/topk_search_query_"+str(k)+".txt")
    # for kw in user_query:
    #     print("Keyword query : "+ kw)
    #     for th in threshold:
    #         print("Threshold : "+str(th))
    #         # threshold search
    #         results = vector_store.threshold_search(kw, th)
    #         answer = ollama_rag_request(llm,results,uq)
    #         save_output_file(ct, kw, th, results, default_filename="output/"+ct+"/threshold_search_keyword_"+str(th)+".txt")
    #     for k in topK:
    #         print("TopK : "+str(k))
    #         # top_k search
    #         results = vector_store.semantic_search(kw, k)
    #         answer = ollama_rag_request(llm,results,uq)
    #         save_output_file(ct, kw, k, results, default_filename="output/"+ct+"/topk_search_keyword_"+str(k)+".txt")




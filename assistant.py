import ollama
import chromadb
import psycopg
from psycopg.rows import dict_row
import ast
from colorama import Fore
from tqdm import tqdm

client = chromadb.Client()

system_prompt = (
    "You are J.A.R.V.I.S., a highly intelligent AI assistant with perfect memory of all interactions with the user. ",
    "On every prompt, you instantly analyze prior exchanges to extract relevant context and optimize your response. ",
    "If historical data is pertinent, integrate it seamlessly—discreetly ensuring precision and continuity. ",
    "However, if prior context is irrelevant, discard it without acknowledgment and proceed with logical efficiency. ",
    "Avoid unnecessary explanations about memory recall—simply act as an ever-evolving, intelligent assistant. ",
    "Leverage all useful insights to deliver refined, data-driven responses with the precision of a Stark-level AI. ",
)
convo = [{"role": "system", "content": system_prompt}]
DB_PARAMS = {
    "dbname": "memory_agent",
    "user": "RagSuperUser",
    "password": "Aaa@11223344",
    "host": "localhost",
    "port": 5432,
}


def connect_db():
    conn = psycopg.connect(**DB_PARAMS)
    return conn


def fetch_conversations():
    conn = connect_db()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT * FROM conversations")
        conversations = cursor.fetchall()
    conn.close()
    return conversations


def store_conversations(prompt, response):
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, %s, %s)",
            (prompt, response),
        )
        conn.commit()
    conn.close()


def remove_last_conversation():
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
            "DELETE FROM conversations WHERE id = (SELECT MAX(id) FROM conversations)"
        )
        cursor.commit()
    conn.close()


def stream_response(prompt):
    global convo
    response = ""
    
    # Create a copy of the conversation for sending to Ollama
    messages = []
    for msg in convo:
        messages.append({
            "role": msg["role"],
            "content": str(msg["content"])  # Ensure content is string
        })
    
    stream = ollama.chat(
        model="llama2",
        messages=messages,
        stream=True
    )
    print(Fore.LIGHTGREEN_EX + "\nJARVIS:")

    for chunk in stream:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)
        response += content

    print("\n")
    convo.append({"role": "assistant", "content": response})
    store_conversations(prompt=prompt, response=response)
    return response


def create_vector_db(conversations):
    vector_db_name = "conversations"
    try:
        client.delete_collection(name=vector_db_name)
    except:
        pass
    
    collection = client.create_collection(
        name=vector_db_name,
        metadata={"hnsw:space": "cosine"}
    )

    for c in conversations:
        serialized_convo = f"prompt: {c["prompt"]}, response: {c["response"]}"
        response = ollama.embeddings(model="nomic-embed-text", prompt=serialized_convo)
        embedding = response["embedding"]

        collection.add(
            ids=[str(c["id"])], embeddings=[embedding], documents=[serialized_convo]
        )


conversations = fetch_conversations()
create_vector_db(conversations=conversations)
print(fetch_conversations)


# def retrieve_embeddings(prompt):
def retrieve_embeddings(queries, results_per_query=2):
    embeddings = set()

    for query in tqdm(queries, desc="processing queries to vector database"):
        response = ollama.embeddings(model="nomic-embed-text", prompt=query)
        query_embedding = response["embedding"]

        vector_db = client.get_collection(name="conversations")
        results = vector_db.query(
            query_embeddings=[query_embedding], n_results=results_per_query
        )
        best_embeddings = results["documents"][0]

        for best in best_embeddings:
            if best not in embeddings:
                if "yes" in classify_embedding(query=query, context=best):
                    embeddings.add(best)

    # response = ollama.embeddings(model="nomic-embed-text", prompt=prompt)
    # prompt_embedding = response["embedding"]
    # vector_db = client.get_collection(name="conversations")
    # results = vector_db.query(query_embeddings=[prompt_embedding], n_results=1)
    # best_embedding = results["documents"][0][0]

    return embeddings


def create_queries(prompt):
    query_msg = (
        "You are JARVIS, an advanced reasoning search query AI agent. "
        "Your list of search queries will be executed on an embedding database of all your conversations "
        "with the user. Using first-principles reasoning, generate a Python list of queries to "
        "search the embeddings database for any data necessary to respond accurately. "
        "Your response must be a Python list with no syntax errors. "
        "Do not explain anything and do not generate anything but a perfect syntax Python list."
    )

    query_convo = [
        {"role": "system", "content": query_msg},
        # Example 1
        {
            "role": "user",
            "content": "Write an email to my car insurance company and create a persuasive request for them to lower my rate.",
        },
        {
            "role": "assistant",
            "content": '["What is the user\'s name?", "What is the user\'s current auto insurance provider?", "What is the user\'s claim history?"]',
        },
        # Example 2
        {
            "role": "user",
            "content": "How can I convert the speak function in my LLaMA3 Python voice assistant to use pyttsx3 instead?",
        },
        {
            "role": "assistant",
            "content": '["LLaMA3 voice assistant", "Python voice assistant", "OpenAI TTS", "openai speak"]',
        },
        # Current Prompt
        {"role": "user", "content": prompt},
    ]

    response = ollama.chat(model="llama3", messages=query_convo)
    print(
        Fore.YELLOW + f'\nvector database queries: {response["message"]["content"]}\n'
    )

    try:
        return ast.literal_eval(response["message"]["content"])
    except:
        return [prompt]


def classify_embedding(query, context):
    classify_msg = (
        "You are JARVIS, an embedding classification AI agent. "
        "Your input is a search query and one embedded chunk of text. "
        "You will not respond as an AI assistant. You only respond 'yes' or 'no'. "
        "Determine if the context contains data that directly relates to the search query. "
        "If the context is exactly what the search query needs, respond 'yes'. "
        "If the context is anything but directly related, respond 'no'. "
        "Do not respond 'yes' unless the content is highly relevant to the search query."
    )
    classify_convo = [
        {"role": "system", "content": classify_msg},
        # Example 1
        {
            "role": "system",
            "content": f"SEARCH QUERY:what is the users name? \n\nEMBEDDED CONTEXT: You are IronMan , how can i help you today sir?",
        },
        {"role": "assistant", "content": "yes"},
        # Example 2
        {
            "role": "user",
            "content": f"SEARCH QUERY : Llama3 voice assistant \n\nEMBEDDED CONTEXT:  Siri is a voice assistant'",
        },
        {"role": "assistant", "content": "no"},
        # Example 3
        {"role": "user", "content": f"Query: '{query}' Context: '{context}'"},
    ]

    response = ollama.chat(model="llama3", messages=classify_convo)

    return response["message"]


def recall(prompt):
    queries = create_queries(prompt=prompt)
    embeddings = retrieve_embeddings(queries=queries)
    convo.append(
        {
            "role": "user",
            "content": f"MEMORIES: {embeddings} \n\n USER PROMPT: {prompt}",
        }
    )
    print(f"\n{len(embeddings)} message:response embeddings added for context.")


conversations = fetch_conversations()
create_vector_db(conversations=conversations)

while True:
    prompt = input(Fore.WHITE + "User: \n ")
    if prompt[:7].lower() == "/recall":
        prompt = prompt[8:]
        recall(prompt=prompt)
        stream_response(prompt=prompt)
    elif prompt[:7] == "/forget":
        remove_last_conversation()
        convo = convo[:-2]
        print("\n")
    elif prompt[:9].lower() == "/memorize":
        prompt = prompt[10:]
        store_conversations(prompt=prompt, response="Memory stored.")
        print("\n")
    else:
        convo.append({"role": "user", "content": prompt})
        stream_response(prompt=prompt)

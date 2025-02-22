"""
J.A.R.V.I.S. - Just A Rather Very Intelligent System
A conversational AI assistant with perfect memory and context awareness.

This module implements a JARVIS-like AI assistant that can:
- Maintain conversations with users
- Store and retrieve conversation history using MySQL
- Use vector embeddings for semantic search
- Provide intelligent memory recall
- Allow explicit memory storage and deletion

Key Features:
- Persistent memory across sessions using MySQL
- Semantic search using ChromaDB vector database
- Intelligent query generation for better recall
- Relevance classification of search results
- Streaming responses for better UX

Usage:
    python assistant_mysql.py

Environment Variables:
    MYSQL_PASSWORD: MySQL root password

Commands:
    /recall [query]  - Search through conversation history
    /forget         - Remove the last conversation
    /memorize [text] - Store information without response
    exit/quit/bye   - End the session
"""

import ollama
import chromadb
import sys
import mysql.connector
from mysql.connector import Error
from mysql.connector import pooling
import os
from dotenv import load_dotenv
from tqdm import tqdm
from colorama import Fore
from contextlib import contextmanager
import json
import ast

# Load environment variables
load_dotenv()

# Initialize system prompt for the assistant
system_prompt = (
    "You are J.A.R.V.I.S., a highly intelligent AI assistant with perfect memory of all interactions with the user. "
    "On every prompt, you instantly analyze prior exchanges to extract relevant context and optimize your response. "
    "If historical data is pertinent, integrate it seamlessly—discreetly ensuring precision and continuity. "
    "However, if prior context is irrelevant, discard it without acknowledgment and proceed with logical efficiency. "
    "Avoid unnecessary explanations about memory recall—simply act as an ever-evolving, intelligent assistant. "
    "Leverage all useful insights to deliver refined, data-driven responses with the precision of a Stark-level AI. "
)

# Initialize conversation history with system prompt
convo = [{"role": "system", "content": system_prompt}]

# Database Configuration
INITIAL_DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': os.getenv('MYSQL_PASSWORD'),
    'ssl_disabled': False,
    'ssl_verify_identity': False,
    'use_pure': True
}

DB_CONFIG = {**INITIAL_DB_CONFIG, 'database': 'conversation_store'}

# Global connection pool
connection_pool = None

def create_database():
    """
    Create the conversation database if it doesn't exist.
    
    This function connects to MySQL without specifying a database and
    creates the conversation_store database if it doesn't exist.
    """
    try:
        conn = mysql.connector.connect(**INITIAL_DB_CONFIG)
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
            print(f"Database '{DB_CONFIG['database']}' created successfully")
        conn.close()
    except Error as e:
        print("Error creating database:", e)
        raise

def initialize_connection_pool():
    """
    Initialize the MySQL connection pool.
    
    Creates a pool of database connections that can be reused across
    the application, improving performance and resource usage.
    """
    global connection_pool
    connection_pool = mysql.connector.pooling.MySQLConnectionPool(
        pool_name="mypool",
        pool_size=5,
        **DB_CONFIG
    )

@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    
    Provides a safe way to acquire and release database connections from the pool.
    Should be used with 'with' statement to ensure proper connection handling.
    
    Yields:
        mysql.connector.connection.MySQLConnection: A database connection from the pool
        
    Raises:
        Error: If connection pool is not initialized
    """
    if connection_pool is None:
        raise Error("Connection pool not initialized")
    
    connection = connection_pool.get_connection()
    try:
        yield connection
    finally:
        connection.close()

# Initialize ChromaDB client
chroma_client = chromadb.Client()

def create_vector_db(conversations):
    """
    Create and populate a vector database from conversations
    """
    vector_db_name = "conversations"
    try:
        chroma_client.delete_collection(name=vector_db_name)
    except:
        pass
    
    collection = chroma_client.create_collection(
        name=vector_db_name,
        metadata={"hnsw:space": "cosine"}
    )

    for c in conversations:
        serialized_convo = f"prompt: {c['prompt']}, response: {c['response']}"
        response = ollama.embeddings(model="nomic-embed-text", prompt=serialized_convo)
        embedding = response["embedding"]

        collection.add(
            ids=[str(c.get('id', 0))], 
            embeddings=[embedding], 
            documents=[serialized_convo]
        )

def create_queries(prompt):
    """
    Generate intelligent search queries for the given prompt.
    
    Uses LLM to analyze the prompt and generate multiple relevant search queries
    that will help find related information in the conversation history.
    
    Args:
        prompt (str): The user's search query
        
    Returns:
        list[str]: List of generated search queries
        
    Note:
        If query generation fails, falls back to using the original prompt
    """
    query_msg = (
        "You are an advanced reasoning search query AI agent. "
        "Generate a Python list of search queries to find relevant information "
        "in the conversation history database. The queries should help find "
        "any context needed to provide an accurate response. "
        "Return only a Python list with no additional text."
    )

    query_convo = [
        {"role": "system", "content": query_msg},
        {"role": "user", "content": prompt},
    ]

    response = ollama.chat(model="phi4", messages=query_convo)
    print(Fore.YELLOW + f'\nGenerating search queries: {response["message"]["content"]}\n')

    try:
        return ast.literal_eval(response["message"]["content"])
    except:
        return [prompt]  # Fallback to original prompt if parsing fails

def classify_embedding(query, context):
    """
    Determine if a context is relevant to a search query.
    
    Uses LLM to analyze whether a piece of context from the conversation
    history is relevant to the current search query.
    
    Args:
        query (str): The search query
        context (str): The potential relevant context
        
    Returns:
        bool: True if context is relevant, False otherwise
        
    Note:
        Returns 'yes' or 'no' in lowercase, meant to be checked with 'in' operator
    """
    classify_msg = (
        "You are a relevance classification agent. Respond only with 'yes' or 'no'. "
        "Determine if the context contains information directly related to the search query. "
        "Respond 'yes' only if the context is highly relevant to answering the query."
    )
    
    classify_convo = [
        {"role": "system", "content": classify_msg},
        {"role": "user", "content": f"Query: '{query}' Context: '{context}'"},
    ]

    response = ollama.chat(model="phi4", messages=classify_convo)
    return response["message"]["content"].lower()

def retrieve_embeddings(queries, results_per_query=2):
    """
    Retrieve relevant embeddings based on input queries
    """
    embeddings = set()

    for query in tqdm(queries, desc="processing queries to vector database"):
        response = ollama.embeddings(model="nomic-embed-text", prompt=query)
        query_embedding = response["embedding"]

        vector_db = chroma_client.get_collection(name="conversations")
        results = vector_db.query(
            query_embeddings=[query_embedding], 
            n_results=results_per_query
        )
        best_embeddings = results["documents"][0]

        for best in best_embeddings:
            if best not in embeddings:
                # Only add the embedding if it's classified as relevant
                if "yes" in classify_embedding(query=query, context=best):
                    embeddings.add(best)
                    print(Fore.CYAN + f"Found relevant context: {best}\n")
                else:
                    print(Fore.RED + f"Filtered out irrelevant context: {best}\n")

    return embeddings

def create_tables():
    """
    Create necessary database tables if they don't exist.
    
    Creates two tables:
    - conversations: Stores the actual conversations
    - embeddings: Stores vector embeddings for semantic search
    
    Tables are linked with a foreign key relationship.
    """
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                # Create conversations table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        prompt TEXT NOT NULL,
                        response TEXT NOT NULL,
                        embedding JSON
                    )
                """)
                
                # Create embeddings table for vector search
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        conversation_id INT,
                        embedding JSON NOT NULL,
                        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                    )
                """)
                connection.commit()
                print("Database tables created successfully")
    except Error as e:
        print("Error creating tables:", e)
        raise

def store_conversation(prompt, response, embedding=None):
    """
    Store conversation and its embedding in MySQL database and vector database
    """
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                # Convert embedding to JSON string if it exists
                embedding_json = json.dumps(embedding) if embedding is not None else None
                
                # Insert conversation
                cursor.execute(
                    "INSERT INTO conversations (timestamp, prompt, response, embedding) VALUES (CURRENT_TIMESTAMP, %s, %s, %s)",
                    (prompt, response, embedding_json)
                )
                conversation_id = cursor.lastrowid
                
                # If we have an embedding, store it separately
                if embedding_json:
                    cursor.execute(
                        "INSERT INTO embeddings (conversation_id, embedding) VALUES (%s, %s)",
                        (conversation_id, embedding_json)
                    )
                connection.commit()

                # Update vector database
                serialized_convo = f"prompt: {prompt}, response: {response}"
                vector_db = chroma_client.get_collection(name="conversations")
                vector_db.add(
                    ids=[str(conversation_id)],
                    embeddings=[embedding] if embedding else None,
                    documents=[serialized_convo]
                )
                
                return conversation_id
    except Error as e:
        print("Error storing conversation:", e)
        return None

def remove_last_conversation():
    """
    Remove the last conversation from both databases and memory
    """
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                # Get the ID of the last conversation
                cursor.execute("SELECT id FROM conversations ORDER BY timestamp DESC LIMIT 1")
                result = cursor.fetchone()
                if not result:
                    print("No conversations to remove")
                    return
                
                last_id = result[0]
                
                # First, remove from embeddings table due to foreign key constraint
                cursor.execute("DELETE FROM embeddings WHERE conversation_id = %s", (last_id,))
                
                # Then remove from conversations table
                cursor.execute("DELETE FROM conversations WHERE id = %s", (last_id,))
                
                connection.commit()
                
                # Remove from vector database
                vector_db = chroma_client.get_collection(name="conversations")
                vector_db.delete(ids=[str(last_id)])
                
                print("Last conversation removed successfully")
    except Error as e:
        print("Error removing last conversation:", e)
        raise

def store_memory(prompt):
    """
    Explicitly store a memory without requiring a response
    """
    try:
        # Generate embedding for the memory
        response = ollama.embeddings(model="nomic-embed-text", prompt=prompt)
        embedding = response["embedding"]
        
        # Store in database with a placeholder response
        store_conversation(prompt=prompt, response="Memory stored.", embedding=embedding)
        print("Memory stored successfully")
    except Exception as e:
        print("Error storing memory:", e)
        raise

def fetch_all_conversations():
    """
    Fetch all conversations from MySQL database
    """
    try:
        with get_db_connection() as connection:
            with connection.cursor(dictionary=True) as cursor:
                cursor.execute("SELECT id, prompt, response FROM conversations")
                conversations = cursor.fetchall()
                print(f"Loaded {len(conversations)} conversations from database")
                return conversations
    except Error as e:
        print("Error fetching conversations:", e)
        return []

def stream_response(prompt):
    """
    Stream the response from the model while maintaining conversation history
    """
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
        model="phi4",
        messages=messages,
        stream=True
    )
    print(Fore.LIGHTBLUE_EX + "\nJ.A.R.V.I.S.:" + Fore.WHITE)

    for chunk in stream:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)
        response += content

    print("\n")
    convo.append({"role": "assistant", "content": response})
    
    # Generate embedding for the conversation
    conversation_text = f"prompt: {prompt}, response: {response}"
    embedding_response = ollama.embeddings(model="nomic-embed-text", prompt=conversation_text)
    embedding = embedding_response["embedding"]
    
    # Store conversation and embedding in database
    store_conversation(prompt=prompt, response=response, embedding=embedding)
    return response

if __name__ == "__main__":
    try:
        # Initialize database and tables
        print("\n=== Initializing Database ===")
        create_database()
        initialize_connection_pool()
        create_tables()
        
        # Load conversation history
        print("\n=== Loading Conversation History ===")
        conversations = fetch_all_conversations()
        
        # Initialize vector database
        print("\n=== Initializing Vector Database ===")
        create_vector_db(conversations)
        print(f"Vector database initialized with {len(conversations)} conversations\n")

        # Display welcome message
        print(Fore.LIGHTBLUE_EX + "\nInitializing J.A.R.V.I.S. - Just A Rather Very Intelligent System")
        print(Fore.LIGHTBLUE_EX + "=" * 60)
        print(Fore.WHITE + "\nAvailable commands:")
        print("- '/recall [query]' to search through conversation history")
        print("- '/forget' to remove the last conversation")
        print("- '/memorize [text]' to store information without response")
        print("- 'exit', 'quit', or 'bye' to end the session")
        print(Fore.LIGHTBLUE_EX + "\nJ.A.R.V.I.S. is ready to assist you, sir.\n" + Fore.WHITE)
        
        # Main conversation loop
        while True:
            user_input = input("\nYou: ").strip()
            
            # Handle exit command
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print(Fore.LIGHTBLUE_EX + "\nGoodbye, sir. Shutting down." + Fore.WHITE)
                break
            
            # Handle recall command
            if user_input.startswith('/recall '):
                query = user_input[8:]  # Remove '/recall ' prefix
                print("\nSearching through conversation history...")
                search_queries = create_queries(query)
                relevant_embeddings = retrieve_embeddings(search_queries)
                
                if relevant_embeddings:
                    print("\nFound relevant conversations:")
                    for emb in relevant_embeddings:
                        print(f"- {emb}")
                else:
                    print("\nNo relevant conversations found.")
                continue
            
            # Handle forget command
            elif user_input == '/forget':
                remove_last_conversation()
                if convo:
                    convo = convo[:-2]  # Remove last user input and assistant response
                print("Last conversation forgotten.")
                continue
            
            # Handle memorize command
            elif user_input.startswith('/memorize '):
                memory = user_input[10:]  # Remove '/memorize ' prefix
                store_memory(memory)
                continue
            
            # Handle normal conversation
            if user_input:
                convo.append({"role": "user", "content": user_input})
                stream_response(user_input)
                
    except KeyboardInterrupt:
        print(Fore.LIGHTBLUE_EX + "\nShutdown requested. Goodbye, sir." + Fore.WHITE)
    except Exception as e:
        print(Fore.RED + f"\nError: {str(e)}" + Fore.WHITE)
        raise

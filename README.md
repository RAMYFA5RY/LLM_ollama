# J.A.R.V.I.S. - Offline AI Assistant

Just A Rather Very Intelligent System - An AI assistant with perfect memory and contextual awareness.

## Overview

JARVIS is a sophisticated AI assistant that maintains conversations with users while providing intelligent memory management and context-aware responses. It uses MySQL for persistent storage and ChromaDB for semantic search capabilities.

## Component Tree

```
JARVIS Assistant
├── Database Management
│   ├── MySQL Operations
│   │   ├── create_database()
│   │   ├── initialize_connection_pool()
│   │   ├── create_tables()
│   │   └── get_db_connection() [context manager]
│   │
│   └── Vector Database (ChromaDB)
│       └── create_vector_db(conversations)
│
├── Memory Management
│   ├── Storage Operations
│   │   ├── store_conversation(prompt, response, embedding)
│   │   └── store_memory(prompt)
│   │
│   ├── Retrieval Operations
│   │   ├── fetch_all_conversations()
│   │   └── retrieve_embeddings(queries, results_per_query)
│   │
│   └── Memory Maintenance
│       └── remove_last_conversation()
│
├── Search Intelligence
│   ├── Query Generation
│   │   └── create_queries(prompt)
│   │
│   └── Relevance Assessment
│       └── classify_embedding(query, context)
│
├── Conversation Handling
│   └── stream_response(prompt)
│       ├── Manages conversation history
│       ├── Streams AI responses
│       └── Stores in databases
│
└── Command Interface
    ├── /recall [query]
    ├── /forget
    ├── /memorize [text]
    └── exit/quit/bye
```

## Data Flow

```
User Input
    │
    ├── Normal Conversation ──────────────────┐
    │   │                                     │
    │   └─> stream_response() ───────────┐    │
    │       │                            │    │
    │       └─> store_conversation() ────┼────┤
    │           │                        │    │
    │           └─> MySQL + ChromaDB <───┘    │
    │                                         │
    ├── /recall [query] ───────────────────┐  │
    │   │                                  │  │
    │   ├─> create_queries() ─────────┐    │  │
    │   │                             │    │  │
    │   └─> retrieve_embeddings() <───┘    │  │
    │       │                              │  │
    │       └─> classify_embedding() ──────┼──┤
    │           │                          │  │
    │           └─> Filtered Results <─────┘  │
    │                                         │
    ├── /forget ──────────────────────────┐   │
    │   │                                 │   │
    │   └─> remove_last_conversation() ───┼───┤
    │       │                             │   │
    │       └─> Update Both DBs <─────────┘   │
    │                                         │
    └── /memorize [text] ─────────────────┐   │
        │                                 │   │
        └─> store_memory() ───────────────┼───┘
            │                             │
            └─> Store in Both DBs <───────┘
```

## Features

- Persistent conversation memory across sessions
- Semantic search through conversation history
- Intelligent query generation for better recall
- Relevance classification of search results
- Real-time streaming responses
- Memory management commands

## Requirements

- Python 3.11+
- MySQL Server
- Ollama with phi4 model installed
- Environment variables in `.env` file:
  ```
  MYSQL_PASSWORD=your_mysql_root_password
  ```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up MySQL and configure environment variables
4. Run the assistant:
   ```bash
   python assistant_mysql.py
   ```

## Commands

- `/recall [query]` - Search through conversation history
- `/forget` - Remove the last conversation
- `/memorize [text]` - Store information without response
- `exit`, `quit`, `bye` - End the session

## Architecture

### Database Structure

#### MySQL Tables
- **conversations**: Stores conversation history
  - id (PRIMARY KEY)
  - timestamp
  - prompt
  - response
  - embedding (JSON)

- **embeddings**: Stores vector embeddings
  - id (PRIMARY KEY)
  - conversation_id (FOREIGN KEY)
  - embedding (JSON)

#### Vector Database
- Uses ChromaDB for semantic search
- Maintains synchronized state with MySQL

## Core Components

### Database Management

\`\`\`python
create_database()
- Creates MySQL database if it doesn't exist
- Initializes basic database structure

initialize_connection_pool()
- Sets up connection pool for better performance
- Manages database connections efficiently

create_tables()
- Creates necessary MySQL tables
- Sets up foreign key relationships
\`\`\`

### Memory Management

\`\`\`python
store_conversation(prompt, response, embedding)
- Stores conversations in MySQL
- Updates vector database
- Maintains data consistency

remove_last_conversation()
- Removes most recent conversation
- Updates both MySQL and vector database
- Handles foreign key constraints

store_memory(prompt)
- Stores explicit memories
- Generates embeddings
- Uses placeholder response
\`\`\`

### Search and Retrieval

\`\`\`python
create_queries(prompt)
- Generates intelligent search queries
- Uses LLM to understand search intent
- Returns list of relevant queries

classify_embedding(query, context)
- Determines relevance of search results
- Returns binary yes/no decision
- Filters out irrelevant results

retrieve_embeddings(queries, results_per_query=2)
- Performs semantic search
- Uses generated queries
- Filters results through classifier
\`\`\`

### Conversation Handling

\`\`\`python
stream_response(prompt)
- Streams AI responses in real-time
- Maintains conversation history
- Stores responses in database

fetch_all_conversations()
- Retrieves all conversations from MySQL
- Used for initializing vector database
- Returns conversations in dictionary format
\`\`\`

## Vector Database Management

\`\`\`python
create_vector_db(conversations)
- Initializes ChromaDB collection
- Generates embeddings for conversations
- Populates vector database
\`\`\`

## Error Handling

- Graceful error handling throughout
- Proper cleanup on shutdown
- Database connection management
- Fallback mechanisms for LLM operations

## Best Practices

1. Always use the connection pool for database operations
2. Maintain consistency between MySQL and vector database
3. Use proper error handling with try/except blocks
4. Clean up resources properly on shutdown

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - Feel free to use and modify as needed.

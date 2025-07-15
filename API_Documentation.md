# RAG System API Documentation

## Overview

This RAG (Retrieval Augmented Generation) system combines a local language model with a vector database to provide contextually relevant answers based on document retrieval. The system uses LangChain framework with FAISS for vector storage and Phi-3 model for text generation.

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Text Input    │───▶│  Embedding Model │───▶│ Vector Database │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
┌─────────────────┐    ┌──────────────────┐              │
│   User Query    │───▶│   RAG Pipeline   │◀─────────────┘
└─────────────────┘    └──────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐
│     Response    │◀───│    LLM (Phi-3)   │
└─────────────────┘    └──────────────────┘
```

## Components

### 1. Language Model (LLM)

#### Configuration
```python
from langchain import LlamaCpp

llm = LlamaCpp(
    model_path="/content/Phi-3-mini-4k-instruct-fp16.gguf",
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=2048,
    seed=42,
    verbose=False
)
```

#### Parameters
- **model_path** (str): Path to the GGUF model file
- **n_gpu_layers** (int): Number of layers to offload to GPU (-1 for all)
- **max_tokens** (int): Maximum number of tokens to generate
- **n_ctx** (int): Context window size
- **seed** (int): Random seed for reproducible outputs
- **verbose** (bool): Enable/disable verbose logging

### 2. Embedding Model

#### Configuration
```python
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name='thenlper/gte-small'
)
```

#### Parameters
- **model_name** (str): HuggingFace model identifier for text embeddings

#### Supported Models
- `thenlper/gte-small`: General Text Embeddings model (384 dimensions)
- Other compatible sentence-transformers models

### 3. Vector Database

#### Setup
```python
from langchain.vectorstores import FAISS

# Create vector database from text documents
db = FAISS.from_texts(texts, embedding_model)
```

#### Methods

##### `from_texts(texts, embedding_model)`
Creates a FAISS vector database from a list of text documents.

**Parameters:**
- `texts` (List[str]): List of text documents to index
- `embedding_model`: Embedding model instance for text vectorization

**Returns:**
- `FAISS`: Vector database instance

##### `as_retriever(search_kwargs)`
Converts the vector database to a retriever for the RAG chain.

**Parameters:**
- `search_kwargs` (dict): Configuration for similarity search
  - `k` (int): Number of documents to retrieve

**Returns:**
- `VectorStoreRetriever`: Retriever instance

### 4. RAG Pipeline

#### Prompt Template
```python
from langchain import PromptTemplate

template = """<|user|>
Relevant information:
{context}

Provide a concise answer the following question using the relevant information provided above:
{question}<|end|>
<|assistant|>"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)
```

#### Chain Configuration
```python
from langchain.chains import RetrievalQA

rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={
        "prompt": prompt
    },
    verbose=True
)
```

#### Parameters
- **llm**: Language model instance
- **chain_type** (str): Type of chain ('stuff', 'map_reduce', 'refine', 'map_rerank')
- **retriever**: Vector database retriever
- **chain_type_kwargs** (dict): Additional arguments for the chain
  - **prompt**: Custom prompt template
- **verbose** (bool): Enable detailed logging

## API Methods

### Query Execution

#### `rag.invoke(question)`
Executes a query against the RAG system.

**Parameters:**
- `question` (str): User question to answer

**Returns:**
- `str`: Generated answer based on retrieved context

**Example:**
```python
# Ask a question
response = rag.invoke('Where was the minimization done?')
print(response)
```

## Usage Examples

### Basic Setup
```python
# 1. Install dependencies
!pip install langchain==0.2.5 faiss-gpu==1.7.2 cohere==5.5.8 langchain-community==0.2.5 rank_bm25==0.2.2 sentence-transformers==3.0.1

# 2. Download model
!wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-fp16.gguf

# 3. Initialize components
from langchain import LlamaCpp
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import RetrievalQA

# 4. Setup LLM
llm = LlamaCpp(
    model_path="/content/Phi-3-mini-4k-instruct-fp16.gguf",
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=2048,
    seed=42,
    verbose=False
)

# 5. Setup embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name='thenlper/gte-small'
)

# 6. Create vector database
texts = ["Your document text here", "Another document", ...]
db = FAISS.from_texts(texts, embedding_model)

# 7. Setup RAG chain
template = """<|user|>
Relevant information:
{context}

Provide a concise answer the following question using the relevant information provided above:
{question}<|end|>
<|assistant|>"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": prompt},
    verbose=True
)

# 8. Query the system
answer = rag.invoke("Your question here")
```

### Text Processing
```python
# Process and clean text documents
text = """Your raw text content here..."""
texts = text.split('.')  # Split by sentences
texts = [t.strip(' \n') for t in texts]  # Clean whitespace
```

## Error Handling

### Common Issues

1. **CUDA/GPU Issues**
   - Ensure CUDA is properly installed
   - Check GPU memory availability
   - Use `n_gpu_layers=0` for CPU-only mode

2. **Model Loading**
   - Verify model file path exists
   - Ensure sufficient disk space
   - Check model file integrity

3. **Memory Issues**
   - Reduce `n_ctx` or `max_tokens`
   - Use smaller embedding models
   - Process documents in batches

## Performance Optimization

### GPU Acceleration
- Set `n_gpu_layers=-1` to use all GPU layers
- Ensure FAISS GPU support is installed
- Use appropriate CUDA-compatible models

### Memory Management
- Adjust context window size based on available memory
- Use efficient embedding models
- Implement document chunking for large texts

### Retrieval Optimization
- Tune the `k` parameter for retrieval count
- Experiment with different similarity thresholds
- Consider hybrid search approaches

## Dependencies

### Required Packages
```
langchain==0.2.5
faiss-gpu==1.7.2
cohere==5.5.8
langchain-community==0.2.5
rank_bm25==0.2.2
sentence-transformers==3.0.1
llama-cpp-python==0.2.78
```

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (optional but recommended)
- Sufficient RAM for model loading
- Internet connection for model downloads

## Security Considerations

- Validate input text to prevent injection attacks
- Implement rate limiting for API endpoints
- Secure model file storage and access
- Monitor resource usage to prevent abuse

## Troubleshooting

### Model Loading Issues
```python
# Check if model file exists
import os
if not os.path.exists("/content/Phi-3-mini-4k-instruct-fp16.gguf"):
    print("Model file not found")
```

### Embedding Model Issues
```python
# Test embedding model
try:
    test_embedding = embedding_model.embed_query("test")
    print(f"Embedding dimension: {len(test_embedding)}")
except Exception as e:
    print(f"Embedding error: {e}")
```

### Vector Database Issues
```python
# Verify vector database creation
if db:
    print(f"Vector database created with {db.index.ntotal} documents")
else:
    print("Failed to create vector database")
```

## Extending the System

### Custom Prompt Templates
```python
# Create domain-specific prompts
custom_template = """<|user|>
Context: {context}
Question: {question}
Instructions: Provide a technical answer with citations.
<|end|>
<|assistant|>"""
```

### Different Chain Types
- **stuff**: Combine all retrieved documents
- **map_reduce**: Process documents separately then combine
- **refine**: Iteratively refine answers
- **map_rerank**: Score and rank responses

### Advanced Retrieval
```python
# Configure advanced retrieval parameters
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 10,
        "score_threshold": 0.8
    }
)
```
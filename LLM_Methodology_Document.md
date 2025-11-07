# Global Methodology Document (GMD)
## LLM-Based Entity Extraction for Unstructured PDF Documents

**Version:** 1.0
**Date:** 07-11-2025
**Author:** Jasir Mohammed

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Why LLMs for PDF Documents](#3-why-llms-for-pdf-documents)
4. [LLM Selection](#4-llm-selection)
5. [RAG Architecture](#5-rag-architecture)
6. [Prompting Techniques](#6-prompting-techniques)
7. [Implementation Guide](#7-implementation-guide)
8. [Entity Extraction Pipeline](#8-entity-extraction-pipeline)
9. [Evaluation and Accuracy](#9-evaluation-and-accuracy)
10. [Cost Management](#10-cost-management)
11. [Error Handling and Fallbacks](#11-error-handling-and-fallbacks)
12. [Performance Optimization](#12-performance-optimization)
13. [Production Deployment](#13-production-deployment)
14. [Alternative Approaches](#14-alternative-approaches)
15. [Future Improvements](#15-future-improvements)

---

## 1. Executive Summary

This document describes the methodology for extracting financial entities from unstructured PDF documents using Large Language Models (LLMs) combined with Retrieval-Augmented Generation (RAG). Unlike structured DOCX files or chat messages, PDF documents contain lengthy, unstructured text that requires advanced natural language understanding.

The approach uses:
- LLMs (GPT-4, Claude 3) for deep semantic understanding
- RAG to handle long documents beyond LLM context limits
- Advanced prompting techniques for accurate entity extraction
- Vector embeddings for semantic search
- Cost optimization strategies for production deployment

Expected accuracy: 85-92% for entity extraction from unstructured PDFs
Processing time: 30-120 seconds per document
Cost: $0.03-0.10 per document

---

## 2. Problem Statement

### 2.1 Challenges with Unstructured PDFs

Unstructured PDF documents present unique challenges:

1. **Length**: Research reports can be 50-200 pages
2. **Complexity**: Mixed content (text, tables, images, charts)
3. **Layout**: Multi-column layouts, headers, footers
4. **Context**: Entities scattered across pages
5. **Ambiguity**: Same term means different things in different contexts
6. **Noise**: Irrelevant content mixed with important data

**Example PDF Structure:**
```
Page 1: Cover page, title, date
Page 2-5: Executive summary
Page 6-50: Main content with scattered financial entities
Page 51-60: Tables and charts
Page 61-70: Appendices
```

The challenge: Extract specific financial entities (Counterparty, Notional, Maturity, etc.) from this unstructured content without reading the entire document manually.

### 2.2 Why Traditional Methods Fail

**Rule-based parsers:**
- Cannot handle varied layouts
- Miss context-dependent entities
- Break when format changes
- High maintenance cost

**Basic NER models:**
- Cannot process long documents
- Miss entities requiring cross-page context
- Poor at understanding financial domain terminology
- Limited to pre-defined entity types

**Why we need LLMs:**
- Can understand context across long documents
- Handle varied formats and layouts
- Understand financial terminology
- Adapt to new entity types without retraining
- Provide explanations for extractions

---

## 3. Why LLMs for PDF Documents

### 3.1 Advantages of LLMs

1. **Deep Understanding**
   - Understand complex sentences and paragraphs
   - Interpret context across pages
   - Handle ambiguous references

2. **Zero-shot Learning**
   - Extract entities without specific training
   - Adapt to new document formats
   - Handle rare or unusual entities

3. **Reasoning Capability**
   - Infer missing information from context
   - Resolve ambiguities using document context
   - Validate extracted entities against business rules

4. **Flexibility**
   - Process multiple entity types simultaneously
   - Adjust extraction strategy based on document type
   - Handle multi-language documents

### 3.2 Limitations of LLMs

1. **Context Window Limits**
   - GPT-4: 8K-128K tokens (depends on model variant)
   - Claude 3: 200K tokens
   - Average PDF: 50-200 pages = 100K-400K tokens
   - Solution: Use RAG to process document in chunks

2. **Cost**
   - GPT-4: $0.03 per 1K input tokens, $0.06 per 1K output tokens
   - Claude 3 Opus: $0.015 per 1K input tokens, $0.075 per 1K output tokens
   - Full document processing expensive
   - Solution: Use RAG to process only relevant sections

3. **Latency**
   - API calls take 5-30 seconds
   - Multiple calls for long documents
   - Solution: Optimize chunking and parallel processing

4. **Consistency**
   - Same input may yield slightly different outputs
   - Temperature setting affects variability
   - Solution: Use temperature=0 for consistent outputs

---

## 4. LLM Selection

### 4.1 Available LLM Options

| Model | Context Window | Cost (Input/Output) | Speed | Accuracy | Best For |
|-------|----------------|---------------------|-------|----------|----------|
| GPT-4 Turbo | 128K tokens | $0.01/$0.03 per 1K | Fast | 90% | General PDFs |
| GPT-4 | 8K tokens | $0.03/$0.06 per 1K | Medium | 92% | Short PDFs |
| Claude 3 Opus | 200K tokens | $0.015/$0.075 per 1K | Medium | 93% | Long PDFs |
| Claude 3 Sonnet | 200K tokens | $0.003/$0.015 per 1K | Fast | 88% | Cost-sensitive |
| Llama 3 70B | 8K tokens | Free (self-hosted) | Slow | 82% | Privacy-critical |
| Mixtral 8x7B | 32K tokens | Free (self-hosted) | Medium | 80% | Budget projects |

### 4.2 Recommended Choice

**Primary: GPT-4 Turbo**
- Good balance of cost, speed, and accuracy
- 128K context handles most documents
- Widely available and well-documented
- Strong performance on financial documents

**Secondary: Claude 3 Sonnet**
- More cost-effective for high-volume processing
- 200K context for very long documents
- Good accuracy for most use cases
- Better at following complex instructions

**Fallback: GPT-3.5 Turbo**
- Much cheaper ($0.0005/$0.0015 per 1K)
- Faster processing
- Acceptable accuracy (75-80%)
- Use when budget is primary concern

### 4.3 Model Configuration

**Recommended Settings:**
```python
llm_config = {
    "model": "gpt-4-turbo-preview",
    "temperature": 0.0,  # Deterministic output
    "max_tokens": 2000,  # Sufficient for entity extraction
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}
```

**Temperature Explanation:**
- Temperature = 0: Most deterministic, consistent outputs
- Temperature = 0.3: Slight variation, still reliable
- Temperature = 0.7: More creative, less consistent
- Temperature = 1.0: Maximum randomness

For entity extraction, always use temperature = 0 for consistency.

---

## 5. RAG Architecture

### 5.1 What is RAG?

Retrieval-Augmented Generation (RAG) combines:
1. **Retrieval**: Find relevant document sections using semantic search
2. **Augmentation**: Add retrieved context to LLM prompt
3. **Generation**: LLM generates answer using the context

**Why RAG?**
- Process documents larger than LLM context window
- Reduce cost by only processing relevant sections
- Improve accuracy by focusing on pertinent information
- Faster processing (don't send entire document to LLM)

### 5.2 RAG Pipeline Overview

```
PDF Document (100 pages)
    |
    v
Step 1: PDF Parsing
    |
    v
Raw Text (200K tokens)
    |
    v
Step 2: Text Chunking
    |
    v
Chunks (400 chunks x 500 tokens each)
    |
    v
Step 3: Embedding Generation
    |
    v
Vector Embeddings (400 x 1536 dimensions)
    |
    v
Step 4: Vector Storage
    |
    v
Vector Database (Pinecone/FAISS)
    |
    v
Step 5: Query Processing
    |
    v
Query: "Find Counterparty and Notional"
    |
    v
Step 6: Semantic Search
    |
    v
Top 5 Relevant Chunks Retrieved
    |
    v
Step 7: Context Assembly
    |
    v
Prompt = Instructions + Retrieved Chunks
    |
    v
Step 8: LLM Processing
    |
    v
Extracted Entities (JSON format)
```

### 5.3 Component Details

#### 5.3.1 PDF Parsing

**Purpose:** Extract text from PDF files while preserving structure.

**Tools:**
- PyPDF2: Basic text extraction
- pdfplumber: Better table handling
- PDFMiner: Layout-aware extraction
- Azure Form Recognizer: Commercial OCR solution

**Recommended: pdfplumber**
```python
import pdfplumber

def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text
            page_text = page.extract_text()
            text += f"\n--- Page {page.page_number} ---\n"
            text += page_text

            # Extract tables separately
            tables = page.extract_tables()
            for table in tables:
                text += "\n[Table]\n"
                for row in table:
                    text += " | ".join([str(cell) for cell in row]) + "\n"

    return text
```

**Handling Scanned PDFs:**
If PDF contains images instead of text:
1. Detect scanned PDF (no extractable text)
2. Apply OCR (Tesseract or AWS Textract)
3. Process extracted text

#### 5.3.2 Text Chunking

**Purpose:** Split long text into manageable pieces for embedding and retrieval.

**Chunking Strategy:**

**Strategy 1: Fixed-size Chunks**
- Chunk size: 500 tokens (approximately 375 words)
- Overlap: 50 tokens (to avoid splitting entities)
- Simple and predictable

**Strategy 2: Semantic Chunks**
- Split by paragraphs or sections
- Maintain semantic coherence
- Better context preservation

**Strategy 3: Sliding Window**
- Fixed-size chunks with overlap
- Ensures entities near boundaries are captured
- Most reliable for entity extraction

**Recommended Implementation:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=2000, overlap=200):
    """
    Split text into chunks with overlap

    Args:
        text: Full document text
        chunk_size: Target size in characters (approx 500 tokens)
        overlap: Overlap between chunks in characters

    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    chunks = splitter.split_text(text)
    return chunks
```

**Chunking Best Practices:**
- Preserve sentence boundaries (don't split mid-sentence)
- Maintain paragraph structure when possible
- Include page numbers in metadata
- Keep tables intact (don't split tables across chunks)

#### 5.3.3 Embedding Generation

**Purpose:** Convert text chunks to numerical vectors for semantic search.

**Embedding Models:**

| Model | Dimensions | Cost | Performance |
|-------|-----------|------|-------------|
| OpenAI text-embedding-ada-002 | 1536 | $0.0001/1K tokens | Best |
| OpenAI text-embedding-3-small | 512 | $0.00002/1K tokens | Good, cheap |
| Cohere embed-english-v3.0 | 1024 | $0.0001/1K tokens | Good |
| Sentence-BERT (free) | 768 | Free | Decent |

**Recommended: OpenAI text-embedding-ada-002**
- High quality semantic understanding
- Good performance on financial text
- Reasonable cost
- Easy integration

**Implementation:**
```python
from openai import OpenAI

client = OpenAI(api_key="your_api_key")

def generate_embeddings(chunks):
    """
    Generate embeddings for text chunks

    Args:
        chunks: List of text strings

    Returns:
        List of embedding vectors
    """
    embeddings = []

    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=chunk
        )
        embedding = response.data[0].embedding
        embeddings.append(embedding)

    return embeddings
```

**Batch Processing:**
For efficiency, process multiple chunks in single API call:
```python
# Process up to 2048 chunks at once
batch_size = 100
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=batch
    )
    embeddings.extend([item.embedding for item in response.data])
```

#### 5.3.4 Vector Storage

**Purpose:** Store embeddings for fast semantic search.

**Vector Database Options:**

**Pinecone (Cloud)**
- Managed service, no maintenance
- Fast query performance
- Automatic scaling
- Cost: $70/month for 1M vectors
- Best for production

**FAISS (Open Source)**
- Facebook AI Similarity Search
- Run locally or self-hosted
- Free but requires infrastructure
- Very fast for millions of vectors
- Best for cost-sensitive projects

**Weaviate (Open Source + Cloud)**
- Open source with managed option
- Built-in ML features
- Good for hybrid search
- Best for complex requirements

**Chroma (Open Source)**
- Lightweight, easy to use
- Good for development and testing
- Limited scale compared to others
- Best for prototypes

**Recommended: Pinecone for Production, FAISS for Development**

**Pinecone Implementation:**
```python
import pinecone

# Initialize Pinecone
pinecone.init(
    api_key="your_api_key",
    environment="us-west1-gcp"
)

# Create index
index_name = "ador-documents"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding size
        metric="cosine"
    )

index = pinecone.Index(index_name)

# Store embeddings
def store_embeddings(document_id, chunks, embeddings):
    """
    Store chunk embeddings in Pinecone

    Args:
        document_id: Unique document identifier
        chunks: List of text chunks
        embeddings: List of embedding vectors
    """
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"{document_id}_chunk_{i}",
            "values": embedding,
            "metadata": {
                "document_id": document_id,
                "chunk_index": i,
                "text": chunk[:500]  # Store first 500 chars for reference
            }
        })

    # Upload in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)
```

**FAISS Implementation (Local):**
```python
import faiss
import numpy as np
import pickle

class FAISSVectorStore:
    def __init__(self, dimension=1536):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine)
        self.metadata = []  # Store chunk text and metadata

    def add_embeddings(self, embeddings, chunks, document_id):
        """Add embeddings to index"""
        embeddings_array = np.array(embeddings).astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_array)

        # Add to index
        self.index.add(embeddings_array)

        # Store metadata
        for i, chunk in enumerate(chunks):
            self.metadata.append({
                "document_id": document_id,
                "chunk_index": i,
                "text": chunk
            })

    def search(self, query_embedding, top_k=5):
        """Search for similar chunks"""
        query_array = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_array)

        scores, indices = self.index.search(query_array, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                results.append({
                    "score": float(score),
                    "metadata": self.metadata[idx]
                })

        return results

    def save(self, path):
        """Save index to disk"""
        faiss.write_index(self.index, f"{path}/faiss.index")
        with open(f"{path}/metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path):
        """Load index from disk"""
        self.index = faiss.read_index(f"{path}/faiss.index")
        with open(f"{path}/metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
```

#### 5.3.5 Semantic Search

**Purpose:** Find most relevant chunks for entity extraction query.

**Query Types:**

**Type 1: General Entity Search**
```python
query = "Extract financial entities including counterparty, notional amount, maturity date, underlying asset"
```

**Type 2: Specific Entity Search**
```python
query = "Find the bank name or counterparty mentioned in this document"
```

**Type 3: Multiple Targeted Searches**
```python
queries = [
    "Find the counterparty or bank name",
    "Find the notional amount or transaction size",
    "Find the maturity date or termination date",
    "Find the underlying asset or security"
]
```

**Implementation:**
```python
def semantic_search(query, top_k=5):
    """
    Search for relevant chunks using semantic similarity

    Args:
        query: Search query string
        top_k: Number of chunks to retrieve

    Returns:
        List of relevant chunks with scores
    """
    # Generate query embedding
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    query_embedding = response.data[0].embedding

    # Search in Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Extract relevant chunks
    relevant_chunks = []
    for match in results.matches:
        relevant_chunks.append({
            "text": match.metadata["text"],
            "score": match.score,
            "chunk_index": match.metadata["chunk_index"]
        })

    return relevant_chunks
```

**Optimization Strategies:**

**1. Hybrid Search (Semantic + Keyword)**
Combine vector search with keyword matching:
```python
def hybrid_search(query, keywords, top_k=5):
    """Combine semantic and keyword search"""
    # Semantic search
    semantic_results = semantic_search(query, top_k=top_k*2)

    # Re-rank based on keyword presence
    scored_results = []
    for result in semantic_results:
        keyword_score = sum(
            1 for keyword in keywords
            if keyword.lower() in result["text"].lower()
        )
        combined_score = result["score"] + (keyword_score * 0.1)
        result["combined_score"] = combined_score
        scored_results.append(result)

    # Sort and return top k
    scored_results.sort(key=lambda x: x["combined_score"], reverse=True)
    return scored_results[:top_k]
```

**2. Multi-query Retrieval**
Use multiple queries to ensure all entity types are covered:
```python
def multi_query_search(entity_queries, top_k_per_query=3):
    """Search using multiple targeted queries"""
    all_chunks = []
    seen_indices = set()

    for query in entity_queries:
        results = semantic_search(query, top_k=top_k_per_query)
        for result in results:
            idx = result["chunk_index"]
            if idx not in seen_indices:
                all_chunks.append(result)
                seen_indices.add(idx)

    return all_chunks
```

**3. Contextual Expansion**
Include surrounding chunks for better context:
```python
def search_with_context(query, top_k=3, context_chunks=1):
    """Retrieve chunks with surrounding context"""
    # Get relevant chunks
    results = semantic_search(query, top_k=top_k)

    # Expand to include adjacent chunks
    expanded_chunks = []
    for result in results:
        chunk_idx = result["chunk_index"]

        # Add previous chunk
        if chunk_idx > 0:
            prev_chunk = get_chunk_by_index(chunk_idx - 1)
            expanded_chunks.append(prev_chunk)

        # Add current chunk
        expanded_chunks.append(result["text"])

        # Add next chunk
        next_chunk = get_chunk_by_index(chunk_idx + 1)
        if next_chunk:
            expanded_chunks.append(next_chunk)

    return "\n\n".join(expanded_chunks)
```

---

## 6. Prompting Techniques

### 6.1 Prompt Engineering Fundamentals

**Effective prompts have:**
1. Clear instructions
2. Context and examples
3. Output format specification
4. Constraints and rules

**Prompt Structure:**
```
[System Message]
You are a financial document analyzer...

[Instructions]
Extract the following entities...

[Examples (Few-shot)]
Example 1: ...
Example 2: ...

[Context]
Document content: ...

[Output Format]
Provide response in JSON format...

[Constraints]
- If entity not found, use null
- Be precise and conservative
```

### 6.2 Zero-shot Prompting

**Definition:** Ask LLM to perform task without examples.

**When to use:**
- Simple extraction tasks
- Well-defined entities
- Cost-sensitive scenarios (shorter prompts)

**Example:**
```python
zero_shot_prompt = """
You are a financial document analyzer. Extract the following entities from the text:

Required Entities:
1. Counterparty: Name of the bank or financial institution
2. Notional: Transaction amount or notional value
3. Maturity: Maturity date or time period
4. Underlying: Underlying asset or security
5. Coupon: Interest rate or coupon percentage
6. Barrier: Barrier level (if applicable)

Document Text:
{document_text}

Instructions:
- Extract only explicitly mentioned entities
- Use exact text from document when possible
- If an entity is not found, use null
- Be precise and conservative in extraction

Provide your response in JSON format:
{{
  "Counterparty": "...",
  "Notional": "...",
  "Maturity": "...",
  "Underlying": "...",
  "Coupon": "...",
  "Barrier": "..."
}}
"""
```

**Advantages:**
- Simple and straightforward
- Shorter prompts (lower cost)
- Works well for common entities

**Disadvantages:**
- Lower accuracy (80-85%)
- May miss subtle patterns
- Less consistent across documents

### 6.3 Few-shot Prompting

**Definition:** Provide examples to guide LLM extraction.

**When to use:**
- Complex entity definitions
- Domain-specific terminology
- Need higher accuracy (90%+)

**Example:**
```python
few_shot_prompt = """
You are a financial document analyzer. Extract entities from financial documents.

Here are examples of correct extractions:

Example 1:
Text: "BANK ABC entered into a EUR 200 million swap transaction with a 5-year maturity. The reference asset is Allianz SE with a 75% barrier."
Extraction:
{{
  "Counterparty": "BANK ABC",
  "Notional": "EUR 200 million",
  "Maturity": "5-year",
  "Underlying": "Allianz SE",
  "Barrier": "75%"
}}

Example 2:
Text: "This termsheet covers a structured note issued by CACIB. Notional amount: USD 50M. The note matures on December 31, 2026 and references the S&P 500 index. Annual coupon of 3.5%."
Extraction:
{{
  "Counterparty": "CACIB",
  "Notional": "USD 50M",
  "Maturity": "December 31, 2026",
  "Underlying": "S&P 500 index",
  "Coupon": "3.5%"
}}

Now extract entities from this document:

Document Text:
{document_text}

Provide your response in the same JSON format. If an entity is not mentioned, use null.
"""
```

**Advantages:**
- Higher accuracy (90-92%)
- Better handling of edge cases
- More consistent outputs

**Disadvantages:**
- Longer prompts (higher cost)
- Need to maintain good examples
- May overfit to example patterns

### 6.4 Chain-of-Thought Prompting

**Definition:** Ask LLM to explain reasoning before providing answer.

**When to use:**
- Complex documents requiring interpretation
- Need to verify extraction logic
- Building trust and explainability

**Example:**
```python
cot_prompt = """
You are a financial document analyzer. Extract entities from the document by following these steps:

Step 1: Read the document carefully
Step 2: Identify mentions of each entity type
Step 3: Extract the exact text for each entity
Step 4: Validate and format the extraction

Document Text:
{document_text}

Entity Types to Extract:
- Counterparty: Bank or financial institution name
- Notional: Transaction amount
- Maturity: Maturity date or period
- Underlying: Reference asset or security

For each entity, provide:
1. Your reasoning (where you found it and why)
2. The extracted value

Example response format:
{{
  "Counterparty": {{
    "reasoning": "Found in first paragraph: 'transaction with BANK ABC'",
    "value": "BANK ABC"
  }},
  "Notional": {{
    "reasoning": "Mentioned in section 2: 'notional amount of EUR 200 million'",
    "value": "EUR 200 million"
  }},
  ...
}}
"""
```

**Advantages:**
- Highest accuracy (92-94%)
- Provides explanations for audit
- Easier to debug failures
- Better at complex reasoning

**Disadvantages:**
- Highest cost (longer outputs)
- Slower processing
- Need to parse reasoning from response

### 6.5 Structured Output Prompting

**Definition:** Use JSON mode or function calling for structured extraction.

**OpenAI Function Calling:**
```python
def extract_entities_with_function_calling(document_text):
    """Use OpenAI function calling for structured extraction"""

    # Define extraction function schema
    entity_extraction_function = {
        "name": "extract_financial_entities",
        "description": "Extract financial entities from document",
        "parameters": {
            "type": "object",
            "properties": {
                "Counterparty": {
                    "type": "string",
                    "description": "Name of bank or financial institution"
                },
                "Notional": {
                    "type": "string",
                    "description": "Transaction amount or notional value"
                },
                "Maturity": {
                    "type": "string",
                    "description": "Maturity date or time period"
                },
                "Underlying": {
                    "type": "string",
                    "description": "Underlying asset or security"
                },
                "Coupon": {
                    "type": "string",
                    "description": "Interest rate or coupon percentage"
                },
                "Barrier": {
                    "type": "string",
                    "description": "Barrier level if applicable"
                }
            },
            "required": ["Counterparty", "Notional", "Maturity", "Underlying"]
        }
    }

    # Call OpenAI with function
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a financial document analyzer specialized in entity extraction."
            },
            {
                "role": "user",
                "content": f"Extract financial entities from this document:\n\n{document_text}"
            }
        ],
        functions=[entity_extraction_function],
        function_call={"name": "extract_financial_entities"},
        temperature=0.0
    )

    # Parse function call response
    function_args = response.choices[0].message.function_call.arguments
    entities = json.loads(function_args)

    return entities
```

**Advantages:**
- Guaranteed JSON format (no parsing errors)
- Type validation built-in
- Cleaner code
- Better error handling

**Disadvantages:**
- Only available for certain models
- Less flexible than free-form prompts
- Harder to include reasoning

### 6.6 Iterative Refinement Prompting

**Definition:** Use multiple LLM calls to refine extraction.

**Process:**
```
Call 1: Extract entities (draft)
    |
    v
Call 2: Validate and correct extraction
    |
    v
Call 3: Final check and formatting
```

**Implementation:**
```python
def iterative_extraction(document_text):
    """Extract entities with iterative refinement"""

    # Step 1: Initial extraction
    draft_prompt = f"""
    Extract financial entities from this document.
    Be liberal in extraction - include anything that might be relevant.

    Document: {document_text}

    Provide entities in JSON format.
    """

    draft_response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": draft_prompt}],
        temperature=0.3  # Allow some creativity
    )

    draft_entities = json.loads(draft_response.choices[0].message.content)

    # Step 2: Validation and refinement
    refine_prompt = f"""
    Review and refine these extracted entities:

    Draft Extraction:
    {json.dumps(draft_entities, indent=2)}

    Original Document:
    {document_text}

    Tasks:
    1. Verify each entity is actually in the document
    2. Remove any incorrect extractions
    3. Fix any formatting issues
    4. Ensure values are precise and accurate

    Provide refined entities in JSON format.
    """

    refined_response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": refine_prompt}],
        temperature=0.0  # Deterministic refinement
    )

    final_entities = json.loads(refined_response.choices[0].message.content)

    return final_entities
```

**Advantages:**
- Highest accuracy (94-96%)
- Self-correcting
- Catches and fixes errors

**Disadvantages:**
- Double the cost (2+ API calls)
- Slower processing
- More complex implementation

### 6.7 Recommended Prompt Template

**Production-Ready Template:**
```python
ENTITY_EXTRACTION_PROMPT = """
You are an expert financial document analyzer with deep knowledge of structured products, derivatives, and financial terminology.

Your task is to extract specific financial entities from the provided document text.

ENTITY DEFINITIONS:

1. Counterparty
   - Definition: Name of the bank, financial institution, or legal entity involved
   - Examples: "BANK ABC", "CACIB", "JP Morgan", "Deutsche Bank"
   - Look for: Party names, issuer information, bank names

2. Notional
   - Definition: Transaction amount, notional value, or principal amount
   - Examples: "EUR 200 million", "USD 50M", "1 million"
   - Look for: Amounts with currency, transaction size, principal

3. Maturity
   - Definition: Maturity date, termination date, or time period
   - Examples: "December 31, 2026", "5 years", "2Y"
   - Look for: Dates, time periods, expiration information

4. Underlying
   - Definition: Reference asset, security, or index
   - Examples: "Allianz SE", "S&P 500", "EUR/USD exchange rate"
   - Look for: Asset names, ticker symbols, reference indices

5. Coupon
   - Definition: Interest rate, coupon rate, or periodic payment
   - Examples: "3.5%", "5% per annum", "EURIBOR + 2%"
   - Look for: Percentages, interest rates, payment rates

6. Barrier
   - Definition: Barrier level, knock-in/knock-out level
   - Examples: "75%", "80% of initial level", "85% barrier"
   - Look for: Percentage thresholds, barrier levels

EXTRACTION RULES:

1. Precision: Extract EXACT text from document - do not paraphrase
2. Completeness: Include all relevant context (e.g., "EUR 200 million" not just "200")
3. Accuracy: Only extract explicitly stated information
4. Null Handling: Use null for entities not found in document
5. Format: Maintain original formatting from document

DOCUMENT TEXT:
---
{document_text}
---

OUTPUT FORMAT:

Provide your extraction in valid JSON format:

{{
  "Counterparty": "extracted value or null",
  "Notional": "extracted value or null",
  "Maturity": "extracted value or null",
  "Underlying": "extracted value or null",
  "Coupon": "extracted value or null",
  "Barrier": "extracted value or null"
}}

IMPORTANT:
- Return ONLY valid JSON, no additional text
- Use double quotes for strings
- Use null (not "null" or empty string) for missing entities
- Ensure JSON is properly formatted and parseable

Begin extraction:
"""
```

---

## 7. Implementation Guide

### 7.1 Complete RAG Implementation

**Full Python Implementation:**

```python
import os
import json
import pdfplumber
from openai import OpenAI
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFEntityExtractor:
    """Complete LLM-based entity extractor for PDF documents"""

    def __init__(self, openai_api_key, pinecone_api_key, pinecone_env):
        """Initialize extractor with API credentials"""
        self.client = OpenAI(api_key=openai_api_key)

        # Initialize Pinecone
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        self.index_name = "ador-pdf-documents"

        # Create or connect to index
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine"
            )
        self.index = pinecone.Index(self.index_name)

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n\n--- Page {page_num} ---\n\n"
                    text += page_text

                # Handle tables
                tables = page.extract_tables()
                for table in tables:
                    text += "\n\n[Table]\n"
                    for row in table:
                        if row:
                            text += " | ".join([str(cell) if cell else "" for cell in row])
                            text += "\n"

        return text

    def chunk_text(self, text):
        """Split text into chunks"""
        chunks = self.text_splitter.split_text(text)
        return chunks

    def generate_embeddings(self, texts):
        """Generate embeddings for text chunks"""
        embeddings = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

        return embeddings

    def store_document(self, document_id, chunks, embeddings):
        """Store document chunks and embeddings"""
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"{document_id}_chunk_{i}",
                "values": embedding,
                "metadata": {
                    "document_id": document_id,
                    "chunk_index": i,
                    "text": chunk
                }
            })

        # Upload in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)

    def retrieve_relevant_chunks(self, query, document_id, top_k=5):
        """Retrieve most relevant chunks for query"""
        # Generate query embedding
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        query_embedding = response.data[0].embedding

        # Search in Pinecone with document filter
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            filter={"document_id": document_id},
            include_metadata=True
        )

        # Extract chunk texts
        chunks = [match.metadata["text"] for match in results.matches]
        return chunks

    def extract_entities(self, document_text):
        """Extract entities using LLM"""
        prompt = f"""
You are an expert financial document analyzer.

Extract the following entities from the document text:

1. Counterparty: Bank or financial institution name
2. Notional: Transaction amount or notional value
3. Maturity: Maturity date or time period
4. Underlying: Reference asset or security
5. Coupon: Interest rate or coupon percentage
6. Barrier: Barrier level (if applicable)

DOCUMENT TEXT:
{document_text}

Provide your response in valid JSON format:
{{
  "Counterparty": "value or null",
  "Notional": "value or null",
  "Maturity": "value or null",
  "Underlying": "value or null",
  "Coupon": "value or null",
  "Barrier": "value or null"
}}

Return ONLY the JSON, no additional text.
"""

        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000
        )

        # Parse JSON response
        response_text = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        entities = json.loads(response_text)
        return entities

    def process_pdf(self, pdf_path, document_id):
        """
        Complete pipeline: Process PDF and extract entities

        Args:
            pdf_path: Path to PDF file
            document_id: Unique identifier for document

        Returns:
            Extracted entities dict
        """
        print(f"Processing PDF: {pdf_path}")

        # Step 1: Extract text from PDF
        print("  1. Extracting text from PDF...")
        text = self.extract_text_from_pdf(pdf_path)
        print(f"     Extracted {len(text)} characters")

        # Step 2: Chunk text
        print("  2. Chunking text...")
        chunks = self.chunk_text(text)
        print(f"     Created {len(chunks)} chunks")

        # Step 3: Generate embeddings
        print("  3. Generating embeddings...")
        embeddings = self.generate_embeddings(chunks)
        print(f"     Generated {len(embeddings)} embeddings")

        # Step 4: Store in vector database
        print("  4. Storing in vector database...")
        self.store_document(document_id, chunks, embeddings)
        print("     Storage complete")

        # Step 5: Retrieve relevant chunks
        print("  5. Retrieving relevant chunks...")
        query = "Extract financial entities including counterparty, notional, maturity, underlying, coupon, barrier"
        relevant_chunks = self.retrieve_relevant_chunks(query, document_id, top_k=5)
        context = "\n\n---\n\n".join(relevant_chunks)
        print(f"     Retrieved {len(relevant_chunks)} relevant chunks")

        # Step 6: Extract entities using LLM
        print("  6. Extracting entities with LLM...")
        entities = self.extract_entities(context)
        print("     Extraction complete")

        return entities


# Usage
def main():
    """Example usage"""
    extractor = PDFEntityExtractor(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_env=os.getenv("PINECONE_ENV")
    )

    # Process PDF
    entities = extractor.process_pdf(
        pdf_path="financial_document.pdf",
        document_id="doc-12345"
    )

    # Display results
    print("\nExtracted Entities:")
    print(json.dumps(entities, indent=2))


if __name__ == "__main__":
    main()
```

### 7.2 Alternative: Using LangChain

**LangChain simplifies RAG implementation:**

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import pinecone

class LangChainPDFExtractor:
    """Simplified PDF extractor using LangChain"""

    def __init__(self, openai_api_key, pinecone_api_key, pinecone_env):
        # Initialize Pinecone
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

        # Initialize components
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            model_name="gpt-4-turbo-preview",
            temperature=0.0,
            openai_api_key=openai_api_key
        )
        self.index_name = "ador-langchain"

    def process_pdf(self, pdf_path):
        """Process PDF with LangChain"""
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        # Create vector store
        vectorstore = Pinecone.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            index_name=self.index_name
        )

        # Create retrieval chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
        )

        # Extract entities
        query = """
        Extract the following financial entities from the document:
        - Counterparty
        - Notional
        - Maturity
        - Underlying
        - Coupon
        - Barrier

        Provide the response in JSON format.
        """

        result = qa_chain.run(query)
        return result
```

---

## 8. Entity Extraction Pipeline

### 8.1 End-to-End Workflow

```
1. PDF Upload
    |
    v
2. Text Extraction (pdfplumber)
    |
    v
3. Text Preprocessing
   - Remove headers/footers
   - Clean formatting
   - Normalize whitespace
    |
    v
4. Text Chunking (500 tokens, 50 token overlap)
    |
    v
5. Embedding Generation (OpenAI text-embedding-ada-002)
    |
    v
6. Vector Storage (Pinecone)
    |
    v
7. Query Formation
   - Multiple targeted queries
   - One per entity type
    |
    v
8. Semantic Search
   - Retrieve top 5 chunks per query
   - Deduplicate chunks
    |
    v
9. Context Assembly
   - Combine retrieved chunks
   - Add instructions and examples
    |
    v
10. LLM Extraction (GPT-4 Turbo)
    |
    v
11. Response Parsing
   - Extract JSON from response
   - Validate format
    |
    v
12. Post-processing
   - Validate entity values
   - Format standardization
   - Confidence scoring
    |
    v
13. Result Storage
   - Save to database
   - Cache for future queries
```

### 8.2 Error Handling at Each Stage

**Stage: PDF Extraction**
- Error: Cannot read PDF (corrupted file)
- Fallback: Try alternative parser (PyPDF2)
- If both fail: Flag for manual processing

**Stage: Embedding Generation**
- Error: API timeout or rate limit
- Fallback: Retry with exponential backoff
- If fails: Use cached embeddings if available

**Stage: Vector Search**
- Error: No relevant chunks found
- Fallback: Use entire document (if within context limit)
- Alternative: Process full document in batches

**Stage: LLM Extraction**
- Error: Invalid JSON response
- Fallback: Retry with stricter prompt
- Alternative: Use regex to extract entities

**Stage: Result Validation**
- Error: Missing required entities
- Fallback: Second extraction pass with targeted queries
- Alternative: Mark for manual review

---

## 9. Evaluation and Accuracy

### 9.1 Metrics

**Primary Metrics:**

1. **Entity-level Accuracy**
   - Correct extractions / Total entities
   - Target: >85%

2. **Document-level Accuracy**
   - Documents with all entities correct / Total documents
   - Target: >75%

3. **Precision**
   - True Positives / (True Positives + False Positives)
   - Target: >90%

4. **Recall**
   - True Positives / (True Positives + False Negatives)
   - Target: >85%

5. **F1 Score**
   - 2 * (Precision * Recall) / (Precision + Recall)
   - Target: >87%

### 9.2 Evaluation Process

**Test Set Creation:**
```python
# Create gold standard test set
test_set = [
    {
        "document_id": "test_001",
        "pdf_path": "test_documents/doc1.pdf",
        "gold_entities": {
            "Counterparty": "BANK ABC",
            "Notional": "EUR 200 million",
            "Maturity": "5 years",
            "Underlying": "Allianz SE",
            "Coupon": "3.5%",
            "Barrier": "75%"
        }
    },
    # ... 100 test documents
]
```

**Evaluation Script:**
```python
def evaluate_extraction(extractor, test_set):
    """Evaluate extractor on test set"""
    results = {
        "correct": 0,
        "total": 0,
        "entity_scores": {}
    }

    for test_case in test_set:
        # Extract entities
        extracted = extractor.process_pdf(
            test_case["pdf_path"],
            test_case["document_id"]
        )

        # Compare with gold standard
        gold = test_case["gold_entities"]

        for entity_name, gold_value in gold.items():
            results["total"] += 1

            # Check if correctly extracted
            extracted_value = extracted.get(entity_name)
            if extracted_value == gold_value:
                results["correct"] += 1
                if entity_name not in results["entity_scores"]:
                    results["entity_scores"][entity_name] = {"correct": 0, "total": 0}
                results["entity_scores"][entity_name]["correct"] += 1

            if entity_name not in results["entity_scores"]:
                results["entity_scores"][entity_name] = {"correct": 0, "total": 0}
            results["entity_scores"][entity_name]["total"] += 1

    # Calculate metrics
    overall_accuracy = results["correct"] / results["total"]

    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    print("\nPer-Entity Accuracy:")
    for entity, scores in results["entity_scores"].items():
        acc = scores["correct"] / scores["total"]
        print(f"  {entity}: {acc:.2%}")

    return results
```

### 9.3 Expected Performance

| Entity Type | Accuracy | Common Issues |
|-------------|----------|---------------|
| Counterparty | 92% | Multiple parties mentioned |
| Notional | 88% | Multiple amounts in document |
| Maturity | 85% | Multiple dates mentioned |
| Underlying | 90% | Complex security descriptions |
| Coupon | 87% | Multiple rates for different periods |
| Barrier | 83% | Not always explicitly labeled |
| **Overall** | **88%** | Context interpretation |

### 9.4 Improving Accuracy

**Strategy 1: Better Prompting**
- Add more examples
- Clarify entity definitions
- Include negative examples

**Strategy 2: Multi-stage Extraction**
- First pass: Broad extraction
- Second pass: Validation and refinement
- Third pass: Confidence scoring

**Strategy 3: Hybrid Approach**
- Use regex for well-defined patterns (ISIN, dates)
- Use LLM for ambiguous entities
- Combine results with confidence weighting

**Strategy 4: Model Fine-tuning**
- Fine-tune GPT-3.5 on labeled financial documents
- More accurate for domain-specific patterns
- Requires 500-1000 labeled examples

---

## 10. Cost Management

### 10.1 Cost Breakdown

**Per Document Cost Estimate:**

| Component | Unit Cost | Quantity | Cost |
|-----------|-----------|----------|------|
| PDF Parsing | Free | 1 | $0.000 |
| Text Chunking | Free | 1 | $0.000 |
| Embedding Generation | $0.0001/1K tokens | 100K tokens | $0.010 |
| Vector Storage | $0.00002/query | 1 query | $0.000 |
| LLM Extraction (GPT-4) | $0.01/1K input + $0.03/1K output | 10K input + 1K output | $0.130 |
| **Total per Document** | | | **$0.140** |

**Cost Optimization Options:**

**Option 1: Use GPT-3.5 Turbo**
- $0.0005/1K input + $0.0015/1K output
- Cost per document: ~$0.010
- Trade-off: 5-10% lower accuracy

**Option 2: Optimize Retrieval**
- Retrieve fewer chunks (3 instead of 5)
- Reduce input tokens to LLM
- Cost savings: 30-40%

**Option 3: Cache Results**
- Cache extracted entities
- Avoid reprocessing same documents
- Cost savings: 100% for cached documents

**Option 4: Use Claude 3 Sonnet**
- $0.003/1K input + $0.015/1K output
- Cost per document: ~$0.045
- Better value for longer documents

### 10.2 Cost Monitoring

```python
class CostTracker:
    """Track API costs for monitoring"""

    def __init__(self):
        self.costs = {
            "embedding": 0.0,
            "llm": 0.0,
            "total": 0.0
        }
        self.token_counts = {
            "embedding_tokens": 0,
            "llm_input_tokens": 0,
            "llm_output_tokens": 0
        }

    def track_embedding_cost(self, token_count):
        """Track embedding generation cost"""
        cost = (token_count / 1000) * 0.0001
        self.costs["embedding"] += cost
        self.costs["total"] += cost
        self.token_counts["embedding_tokens"] += token_count

    def track_llm_cost(self, input_tokens, output_tokens, model="gpt-4-turbo"):
        """Track LLM API cost"""
        if model == "gpt-4-turbo":
            input_cost = (input_tokens / 1000) * 0.01
            output_cost = (output_tokens / 1000) * 0.03
        elif model == "gpt-3.5-turbo":
            input_cost = (input_tokens / 1000) * 0.0005
            output_cost = (output_tokens / 1000) * 0.0015

        cost = input_cost + output_cost
        self.costs["llm"] += cost
        self.costs["total"] += cost
        self.token_counts["llm_input_tokens"] += input_tokens
        self.token_counts["llm_output_tokens"] += output_tokens

    def get_summary(self):
        """Get cost summary"""
        return {
            "costs": self.costs,
            "token_counts": self.token_counts,
            "average_cost_per_document": self.costs["total"] / max(1, self.document_count)
        }
```

### 10.3 Budget Management Strategies

**Strategy 1: Tiered Processing**
```python
def process_with_budget(pdf_path, budget_tier):
    """Process document based on budget tier"""
    if budget_tier == "premium":
        # Use GPT-4 with full RAG
        model = "gpt-4-turbo"
        top_k = 5
    elif budget_tier == "standard":
        # Use GPT-3.5 with reduced retrieval
        model = "gpt-3.5-turbo"
        top_k = 3
    elif budget_tier == "economy":
        # Use cheaper model, fewer chunks
        model = "gpt-3.5-turbo"
        top_k = 2

    # Process accordingly
    # ...
```

**Strategy 2: Smart Caching**
- Cache at multiple levels:
  - Document embeddings (30 days)
  - Extraction results (90 days)
  - Common queries (7 days)

**Strategy 3: Batch Processing**
- Process multiple documents in single session
- Reuse embeddings model loaded in memory
- Amortize API overhead

**Strategy 4: Fallback Cascade**
```python
def extract_with_fallback(document, max_cost=0.20):
    """Try extraction methods in order of cost"""
    cost = 0.0

    # Try 1: Cached result
    cached = get_cached_result(document.id)
    if cached:
        return cached, 0.0

    # Try 2: Regex patterns (free)
    regex_result = extract_with_regex(document)
    if is_complete(regex_result):
        return regex_result, 0.0

    # Try 3: GPT-3.5 (cheap)
    if cost + 0.01 <= max_cost:
        gpt35_result = extract_with_llm(document, model="gpt-3.5")
        cost += 0.01
        if is_complete(gpt35_result):
            return gpt35_result, cost

    # Try 4: GPT-4 (expensive, best accuracy)
    if cost + 0.14 <= max_cost:
        gpt4_result = extract_with_llm(document, model="gpt-4")
        cost += 0.14
        return gpt4_result, cost

    # Budget exceeded
    return None, cost
```

---

## 11. Error Handling and Fallbacks

### 11.1 Common Failure Scenarios

**Scenario 1: API Rate Limit**
```python
def call_with_retry(func, max_retries=3):
    """Retry with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait_time = (2 ** attempt) * 5  # 5, 10, 20 seconds
            print(f"Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time)
```

**Scenario 2: Invalid JSON Response**
```python
def parse_llm_response(response_text):
    """Parse LLM response with fallback parsing"""
    # Try direct JSON parse
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown
    if "```json" in response_text:
        json_text = response_text.split("```json")[1].split("```")[0]
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

    # Try regex extraction
    json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Give up
    raise ValueError("Could not parse JSON from LLM response")
```

**Scenario 3: No Relevant Chunks Found**
```python
def extract_with_fallback_strategy(document_id, text):
    """Multiple strategies if RAG fails"""
    # Strategy 1: RAG with semantic search
    try:
        chunks = retrieve_relevant_chunks(document_id, top_k=5)
        if chunks:
            context = "\n\n".join(chunks)
            return extract_entities(context)
    except Exception as e:
        print(f"RAG failed: {e}")

    # Strategy 2: Process full text if within limits
    if len(text) < 100000:  # ~25K tokens
        return extract_entities(text)

    # Strategy 3: Process in sliding windows
    windows = create_sliding_windows(text, window_size=20000, overlap=2000)
    all_entities = []
    for window in windows:
        entities = extract_entities(window)
        all_entities.append(entities)

    # Merge results from all windows
    return merge_entity_results(all_entities)
```

**Scenario 4: Partial Extraction**
```python
def handle_partial_extraction(entities):
    """Handle incomplete entity extraction"""
    required_entities = ["Counterparty", "Notional", "Maturity"]
    missing = [e for e in required_entities if not entities.get(e)]

    if missing:
        print(f"Warning: Missing required entities: {missing}")

        # Attempt targeted extraction for missing entities
        for entity in missing:
            targeted_result = extract_specific_entity(entity)
            if targeted_result:
                entities[entity] = targeted_result

    return entities, missing
```

### 11.2 Graceful Degradation

When full extraction fails, provide partial results:

```python
def extract_with_degradation(pdf_path):
    """Extract with progressive fallback"""
    result = {
        "status": "success",
        "method": None,
        "entities": {},
        "warnings": []
    }

    try:
        # Method 1: Full RAG + GPT-4 (best quality)
        result["entities"] = extract_with_rag_gpt4(pdf_path)
        result["method"] = "rag_gpt4"
        return result
    except Exception as e:
        result["warnings"].append(f"GPT-4 failed: {str(e)}")

    try:
        # Method 2: RAG + GPT-3.5 (good quality, cheaper)
        result["entities"] = extract_with_rag_gpt35(pdf_path)
        result["method"] = "rag_gpt35"
        result["status"] = "partial"
        return result
    except Exception as e:
        result["warnings"].append(f"GPT-3.5 failed: {str(e)}")

    try:
        # Method 3: Regex-only (basic quality, free)
        result["entities"] = extract_with_regex(pdf_path)
        result["method"] = "regex"
        result["status"] = "degraded"
        return result
    except Exception as e:
        result["warnings"].append(f"Regex failed: {str(e)}")

    # All methods failed
    result["status"] = "failed"
    result["method"] = "none"
    return result
```

---

## 12. Performance Optimization

### 12.1 Latency Optimization

**Current Bottlenecks:**
1. PDF parsing: 5-10 seconds
2. Embedding generation: 10-20 seconds
3. Vector search: 0.5-1 second
4. LLM extraction: 15-30 seconds
5. **Total: 30-60 seconds per document**

**Optimization Strategies:**

**1. Parallel Processing**
```python
import concurrent.futures

def parallel_chunk_processing(chunks):
    """Process multiple chunks in parallel"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Generate embeddings in parallel
        embedding_futures = [
            executor.submit(generate_embedding, chunk)
            for chunk in chunks
        ]
        embeddings = [f.result() for f in embedding_futures]

    return embeddings
```

**2. Batch API Calls**
```python
# Instead of individual calls
for chunk in chunks:
    embedding = generate_embedding(chunk)  # Slow

# Use batch calls
embeddings = generate_embeddings_batch(chunks)  # Fast
```

**3. Caching**
```python
import hashlib
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embedding(text):
    """Cache embeddings for identical text"""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def get_embedding_with_cache(text):
    """Get embedding with caching"""
    text_hash = hashlib.md5(text.encode()).hexdigest()

    # Check cache
    cached = redis_client.get(f"embedding:{text_hash}")
    if cached:
        return json.loads(cached)

    # Generate and cache
    embedding = generate_embedding(text)
    redis_client.setex(
        f"embedding:{text_hash}",
        86400,  # 24 hour TTL
        json.dumps(embedding)
    )

    return embedding
```

**4. Streaming**
```python
def stream_llm_extraction(context):
    """Stream LLM response for faster perceived performance"""
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": context}],
        stream=True
    )

    full_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response += content
            print(content, end="", flush=True)  # Real-time output

    return full_response
```

**5. Precomputation**
```python
def preprocess_document_async(pdf_path, document_id):
    """Pre-process document before user requests extraction"""
    # Step 1: Extract and chunk (done immediately on upload)
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    # Step 2: Generate and store embeddings (background job)
    embeddings = generate_embeddings(chunks)
    store_document(document_id, chunks, embeddings)

    # Now extraction will be instant (just LLM call)
```

### 12.2 Throughput Optimization

**Goal: Process 100+ documents per hour**

**Architecture:**
```
Document Queue
     |
     v
Worker Pool (10 workers)
     |
     +---> Worker 1 --> Extract --> Results
     +---> Worker 2 --> Extract --> Results
     +---> Worker 3 --> Extract --> Results
     |     ...
     +---> Worker 10 -> Extract --> Results
```

**Implementation:**
```python
from celery import Celery
from redis import Redis

app = Celery('pdf_processor', broker='redis://localhost:6379')

@app.task
def process_pdf_task(pdf_path, document_id):
    """Celery task for PDF processing"""
    extractor = PDFEntityExtractor(...)
    entities = extractor.process_pdf(pdf_path, document_id)

    # Store results
    store_results(document_id, entities)

    # Notify user
    notify_completion(document_id)

    return entities

# Submit documents for processing
def submit_documents(pdf_paths):
    """Submit multiple documents for parallel processing"""
    tasks = []
    for pdf_path in pdf_paths:
        document_id = generate_id()
        task = process_pdf_task.delay(pdf_path, document_id)
        tasks.append(task)

    return tasks
```

**Scaling Strategy:**
- Start with 10 workers
- Monitor queue depth
- Auto-scale workers based on load
- Target: Queue depth < 50 documents

---

## 13. Production Deployment

### 13.1 System Requirements

**Compute:**
- CPU: 4 cores minimum, 8 cores recommended
- RAM: 16GB minimum, 32GB recommended
- Storage: 100GB for documents + vector database

**Dependencies:**
- Python 3.10+
- pdfplumber
- OpenAI Python SDK
- Pinecone client
- Redis (for caching)
- PostgreSQL (for metadata)

### 13.2 Environment Configuration

```bash
# .env file
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo-preview
PINECONE_API_KEY=...
PINECONE_ENV=us-west1-gcp
PINECONE_INDEX=ador-pdf-documents
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost/ador
```

### 13.3 Monitoring

**Key Metrics:**
- Processing time per document
- API costs per document
- Success rate
- Error rate by type
- Queue depth

**Alerts:**
- Processing time > 120 seconds
- Error rate > 5%
- API costs > budget threshold
- Queue depth > 100

### 13.4 Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_extraction.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def process_with_logging(pdf_path, document_id):
    """Process with comprehensive logging"""
    logger.info(f"Starting processing: {document_id}")

    try:
        # Extract text
        logger.info(f"Extracting text from: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        logger.info(f"Extracted {len(text)} characters")

        # Chunk
        chunks = chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks")

        # Generate embeddings
        embeddings = generate_embeddings(chunks)
        logger.info(f"Generated embeddings")

        # Extract entities
        entities = extract_entities(context)
        logger.info(f"Extracted entities: {list(entities.keys())}")

        logger.info(f"Completed processing: {document_id}")
        return entities

    except Exception as e:
        logger.error(f"Error processing {document_id}: {str(e)}", exc_info=True)
        raise
```

---

## 14. Alternative Approaches

### 14.1 Using Local LLMs

**Option: Llama 3 70B (Self-hosted)**

**Advantages:**
- No API costs after setup
- Full data privacy
- Unlimited requests

**Disadvantages:**
- Requires powerful hardware (A100 GPUs)
- Lower accuracy than GPT-4 (82% vs 90%)
- Slower inference
- Maintenance overhead

**Implementation:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalLLMExtractor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B")
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-70B",
            device_map="auto",
            load_in_8bit=True
        )

    def extract_entities(self, context):
        """Extract using local LLM"""
        prompt = f"Extract financial entities from:\n\n{context}\n\nEntities:"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=500)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return self.parse_response(response)
```

### 14.2 Hybrid LLM + NER Approach

**Combine strengths of both approaches:**

```python
def hybrid_extraction(pdf_path):
    """Use NER for structured entities, LLM for ambiguous ones"""

    # Step 1: NER for well-defined entities
    ner_entities = {
        "ISIN": extract_isin_with_regex(text),
        "Dates": extract_dates_with_ner(text),
        "Amounts": extract_amounts_with_ner(text)
    }

    # Step 2: LLM for context-dependent entities
    llm_entities = {
        "Counterparty": extract_counterparty_with_llm(text),
        "Underlying": extract_underlying_with_llm(text)
    }

    # Step 3: Merge results
    entities = {**ner_entities, **llm_entities}

    return entities
```

**Advantages:**
- Lower cost (less LLM usage)
- Faster (NER is quicker)
- Higher accuracy for structured fields

### 14.3 Fine-tuned Models

**Fine-tune GPT-3.5 on financial documents:**

**Process:**
1. Collect 500-1000 labeled PDFs
2. Format as training data
3. Fine-tune via OpenAI API
4. Deploy custom model

**Expected Results:**
- Accuracy improvement: 78% -> 87%
- Cost reduction: 50% vs GPT-4
- Faster inference

**Cost:**
- Training: $100-500 one-time
- Inference: Same as GPT-3.5 base model

---

## 15. Future Improvements

### 15.1 Short-term (1-3 months)

**1. Multi-modal Processing**
- Extract information from charts and tables
- Use GPT-4 Vision for image understanding
- Combine text and visual data

**2. Active Learning**
- Flag low-confidence extractions
- Collect user corrections
- Retrain models with feedback

**3. Entity Validation**
- Cross-reference with external databases
- Validate ISINs, company names
- Flag inconsistencies

### 15.2 Medium-term (3-6 months)

**1. Custom Entity Types**
- Allow users to define new entity types
- Dynamic prompt generation
- No code changes required

**2. Multi-document Analysis**
- Compare entities across documents
- Identify patterns and anomalies
- Generate summaries

**3. Real-time Processing**
- Reduce latency to < 10 seconds
- Stream results as they're extracted
- Progressive enhancement UI

### 15.3 Long-term (6-12 months)

**1. Specialized Models**
- Fine-tune on large corpus of financial PDFs
- Achieve 95%+ accuracy
- Support multiple languages

**2. Autonomous Agents**
- LLM agents that can navigate complex documents
- Multi-step reasoning for entity extraction
- Self-correction and validation

**3. Integration Ecosystem**
- Direct integrations with trading systems
- Automated workflow triggers
- API marketplace for extensions

---

**End of Document**

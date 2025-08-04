# HDC Markdown Transformer - Current Architecture

## Overview

The HDC Markdown Transformer is a production-ready system that encodes Markdown documents into hyperdimensional vectors (HDC) for semantic compression and faithful reconstruction. The system uses a **dual HDC encoding** approach that separates content and positional information, enabling precise order recovery through HDC binding/unbinding operations combined with LLM-powered reconstruction.

## Core Innovation: Dual HDC Pipeline

Unlike traditional HDC approaches that encode only semantic content, this system implements a **dual encoding strategy**:

- **Content Vector**: Semantic representation of all words (bundling of word vectors)
- **Position Vector**: Positional information encoding (bundling of position vectors)  
- **Pair Vectors**: HDC binding of word-position associations for order recovery

This enables both semantic compression and faithful reconstruction with preserved word order.

## Architecture Overview

```mermaid
graph TD
    A[Markdown Document] -->|Preprocessing| B[Token Extraction]
    B -->|Vocabulary Check| C[HDC Dictionary Coverage]
    C -->|Dual Encoding| D[HDC Encoder]
    D -->|Content + Position| E[Dual Hypervectors]
    E -->|Binding| F[Pair Vectors Generation]
    F -->|Storage| G[Vector Files (.npy)]
    
    G -->|Loading| H[Vector Reconstruction]
    H -->|Content Search| I[Similarity Search Engine]
    H -->|Position Search| J[Position Database]
    I -->|Word Candidates| K[HDC Unbinding]
    J -->|Position Candidates| K
    K -->|Order Recovery| L[Ordered Word-Position Pairs]
    L -->|LLM Reconstruction| M[Gemini LLM]
    M -->|Generation| N[Reconstructed Markdown]
```

## System Components

### 1. Preprocessing Pipeline
- **Tokenizer**: SpaCy-based tokenization with normalization
- **Vocabulary Coverage**: Validation against 20,000-word HDC dictionary
- **Document Structure**: Preserves Markdown formatting context

### 2. HDC Dictionary (ItemMemory)
- **Dimension**: 10,000-dimensional bipolar vectors (±1)
- **Vocabulary**: 20,000 English words from curated word lists
- **Storage**: Cached in `item_memory.npz` for performance
- **Vector Database**: In-memory storage for similarity search

### 3. Dual HDC Encoder
- **Content Encoding**: Bundling (sum) of all word vectors in document
- **Position Encoding**: Bundling of position vectors (0 to document_length-1)
- **Pair Generation**: HDC binding `word_vector * position_vector` for each token
- **Output**: Three vectors (content, position, pairs) saved as `.npy` files

### 4. Storage System
- **Vector Files**: 
  - `encoded_{n}_content.npy`: Content hypervector
  - `encoded_{n}_position.npy`: Position hypervector  
  - `encoded_{n}_pairs.npy`: Array of word-position bindings
- **Metadata**: Token count encoded in filename for reconstruction that's all

### 5. Reconstruction Pipeline

#### Phase 1: Similarity Search
- **Content Search**: K-NN search on content vector → word candidates
- **Position Search**: K-NN search on position vector → position candidates
- **Configuration**: k=token_count for precise candidate retrieval

#### Phase 2: HDC Unbinding (Order Recovery)
- **Document Vector**: Sum of all pair vectors
- **Unbinding Process**: `document_vector * position_vector ≈ word_vector`
- **Order Deduction**: Test each word-position combination for best similarity
- **Output**: Ordered list of word-position pairs with confidence scores

#### Phase 3: LLM Reconstruction
- **Provider**: Google Gemini Pro
- **Input**: Ordered word candidates with positions and similarity scores
- **Process**: Contextual markdown generation from structured candidates
- **Output**: Faithful reconstruction of original document

## Key Technical Features

### HDC Operations
- **Binding**: Element-wise multiplication for association creation
- **Bundling**: Element-wise addition for set representation
- **Unbinding**: Multiplication with inverse for information retrieval
- **Similarity**: Cosine similarity for vector comparison

### Performance Optimizations
- **Caching**: Dictionary and vector database caching
- **Memory Management**: In-memory vector database for speed
- **Batch Processing**: Efficient vector operations using NumPy
- **Adaptive Search**: Dynamic k-value based on document length

### Quality Assurance
- **Vocabulary Coverage**: 100% coverage validation before encoding
- **Order Validation**: HDC unbinding verification for position accuracy
- **Reconstruction Metrics**: Similarity scoring and confidence thresholds
- **Fallback Mechanisms**: Graceful degradation when unbinding fails

## Configuration

The system supports flexible configuration through YAML:

```yaml
hdc:
  dimension: 10000
  random_seed: 42
  
tokenizer:
  type: spacy
  
vector_database:
  type: memory
  
llm:
  provider: gemini
  model: gemini-pro
  
reconstruction:
  max_candidates: 20
  min_confidence_threshold: 0.1
```

## Current Status

After comprehensive code purge and optimization:

- **Simplified Architecture**: Removed external vector databases, evaluation modules
- **Dual HDC Pipeline**: Fully functional encoding and reconstruction
- **In-Memory Storage**: Single vector database backend for performance
- **LLM Integration**: Gemini-powered reconstruction with contextual generation
- **Production Ready**: Complete CLI interface with logging and error handling

## Future Integration

This HDC Markdown Transformer serves as a foundation component for the broader **OpenDataHive** ecosystem, where it will be extended to:

- Support multiple file formats beyond Markdown
- Integrate with universal `.odhc` vector format
- Enable distributed dictionary sharing
- Provide standardized compression/decompression APIs

The system demonstrates the viability of HDC-based semantic compression with faithful reconstruction, paving the way for universal vector-based data interchange.

Cli commands for current 04 08 2025:

 python cli.py --encode-dual "md de test\readme.md" --config config.yaml --log pipeline_debug.log

 python cli.py --reconstruct-dual --content-vector "encoded_vectors\encoded_12_content.npy" --position-vector "encoded_vectors\encoded_12_position.npy" -o "reconstructed_output.md" --log pipeline_debug.log
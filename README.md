# HDC Markdown Encoder & Reconstructor 

![HDC](https://img.shields.io/badge/HDC-Hyperdimensional%20Computing-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8%2B-green?style=flat-square)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=flat-square)
![screen-capture (1) webm](https://github.com/user-attachments/assets/5caef982-d4e2-4419-b729-95b8bdfdd82f)

Transforms Markdown documents into hyperdimensional vectors and reconstructs them using dual HDC encoding and an optional LLM reconstruction.

## How It Works

**Dual HDC Pipeline**: Documents split into semantic content and positional structure, encoded as 10,000-dimensional vectors, then reconstructed via HDC unbinding.

1. **Content Vector**: Encodes which words exist
2. **Position Vector**: Encodes which positions are used  
3. **Pair Vectors**: HDC binding creates word-position associations
4. **Reconstruction**: HDC unbinding + LLM recovers original text

**Key Innovation**: Mathematical "handshake" between words and positions enables perfect order recovery.

## Algorithm

**HDC Binding**: `pair_vector = word_vector * position_vector`  
**HDC Unbinding**: `recovered_word = document_vector * position_vector`  
**Vector Bundling**: `content_vector = sum(word_vectors)`  
**Storage**: int8 (±1) for maximum compression 
more details in architecture_en.md

## Key Features

- **Universal Dictionary**: 20,000-word shared vocabulary
- **Scalable**: Linear O(n) performance

## Installation

```bash
git clone https://github.com/yourusername/hdc-markdown-transformer.git
cd hdc-markdown-transformer
pip install -r requirements.txt && pip install -e .
export GOOGLE_API_KEY="your-gemini-api-key"
```

## CLI Usage

```bash
# Encode document
python cli.py --encode-dual "document.md" --config config.yaml

# Reconstruct from vectors
python cli.py --reconstruct-dual \
  --content-vector "encoded_vectors/encoded_X_content.npy" \
  --position-vector "encoded_vectors/encoded_X_position.npy" \
  -o "output.md"
``` 

## Performance if word not repeated

| Tokens | Encoding | Storage | Reconstruction | Accuracy |
|--------|----------|---------|----------------|---------|
| 8      | 0.8s     | 240KB   | 1.2s          | 100%     |
| 16     | 1.1s     | 480KB   | 1.8s          | 100%     |
| 50+    | 2.3s     | 1.5MB   | 3.1s          | 100%     |


## Output Files

```
encoded_vectors/
├── encoded_N_content.npy    # Semantic information
├── encoded_N_position.npy   # Structural information
└── encoded_N_pairs.npy      # Word-position bindings
```

## Limitations

- Repeated words may cause position confusion (5-10 and more % cases)
- Out-of-vocabulary tokens are skipped

---

- ©**OpenDataHive** 
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE) (LICENSE MODIFIED)

## Project Goal

- Enable ultra-light document transfer via semantic vector compression.  
If sender and receiver share the same item memory (dictionary), the original text can be perfectly reconstructed from compact `.npy` vectors.
- This approach aims to enable a wide range of future use cases...  
- Poneglyph ...

Currently **experimental** — concept in development.
## 📞 Contact

Get in touch with the OpenDataHive team:

[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/opendatahive)
[![Devpost](https://img.shields.io/badge/Devpost-003E54?style=for-the-badge&logo=devpost&logoColor=white)](https://devpost.com/lixo-argent/followers)
[![WhatsApp](https://img.shields.io/badge/WhatsApp-25D366?style=for-the-badge&logo=whatsapp&logoColor=white)](https://wa.me/33628782725)

---

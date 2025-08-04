import numpy as np
from pathlib import Path

# Génère un dictionnaire HDC à partir d'une liste de mots (vocab.txt)
def generate_hdc_dictionary(vocab_file, dimension=10000, seed=42, batch_size=500, output_file="item_memory.npz"):
    np.random.seed(seed)
    with open(vocab_file, encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]
    n_words = len(vocab)
    print(f"Generating HDC dictionary for {n_words} words (dim={dimension})...")
    
    # Génération progressive par batchs
    vectors = np.empty((n_words, dimension), dtype=np.float32)
    for i in range(0, n_words, batch_size):
        batch_vocab = vocab[i:i+batch_size]
        batch_vectors = np.random.choice([-1, 1], size=(len(batch_vocab), dimension)).astype(np.float32)
        vectors[i:i+len(batch_vocab)] = batch_vectors
        print(f"Generated vectors for words {i} to {i+len(batch_vocab)-1}")
    # Sauvegarde
    np.savez_compressed(output_file, vocab=vocab, vectors=vectors)
    print(f"Dictionary saved to {output_file} (shape={vectors.shape})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate HDC dictionary from vocab list.")
    parser.add_argument("vocab_file", help="Path to vocab.txt file")
    parser.add_argument("--dimension", type=int, default=10000, help="HDC vector dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for generation")
    parser.add_argument("--output", default="item_memory.npz", help="Output .npz file")
    args = parser.parse_args()
    generate_hdc_dictionary(args.vocab_file, args.dimension, args.seed, args.batch_size, args.output)

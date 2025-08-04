import argparse
import logging
import asyncio
import numpy as np
from hdc_markdown_transformer.pipeline import HDCMarkdownTransformer


def main():
    parser = argparse.ArgumentParser(description="HDC Markdown Transformer CLI")
    parser.add_argument("input", type=str, nargs="?", help="Chemin du fichier markdown d'entrée")
    parser.add_argument("-o", "--output", type=str, help="Chemin du fichier markdown ou hypervecteur de sortie")
    parser.add_argument("-c", "--config", type=str, help="Chemin du fichier de configuration YAML (optionnel)")
    parser.add_argument("--encode-dual", action="store_true", help="Générer et sauvegarder les hypervecteurs HDC dual (content + position)")
    parser.add_argument("--reconstruct-dual", action="store_true", help="Reconstruire depuis des hypervecteurs dual (content + position)")
    parser.add_argument("--content-vector", type=str, help="Chemin du vecteur de contenu (pour reconstruction dual)")
    parser.add_argument("--position-vector", type=str, help="Chemin du vecteur de position (pour reconstruction dual)")
    parser.add_argument("--log", type=str, default="INFO", help="Niveau de log (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--k", type=int, default=200, help="Nombre de candidats à retourner lors de la recherche de similarité (k)")
    parser.add_argument("--similarity-threshold", type=float, default=0.0, help="Seuil de similarité minimal pour retenir un candidat")
    args = parser.parse_args()

    # Gestion améliorée du logging : si --log est un chemin de fichier, écrire dans ce fichier + console
    import os
    if args.log and args.log.lower().endswith('.log'):
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s:%(name)s:%(message)s",
            handlers=[
                logging.FileHandler(args.log, mode='w', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO))

    # Initialiser le pipeline
    transformer = HDCMarkdownTransformer(config_path=args.config)
    
    # Encodage dual
    if args.encode_dual:
        if not args.input:
            parser.error("--encode-dual requires input file")
        
        with open(args.input, "r", encoding="utf-8") as f:
            markdown_content = f.read()
        
        transformer.encode_only_dual(markdown_content)
        return
    
    # Reconstruction dual à partir d'hypervecteurs dual
    elif args.reconstruct_dual:
        if not args.content_vector or not args.position_vector:
            parser.error("--reconstruct-dual requires both --content-vector and --position-vector")
        
        # Load vectors
        content_vector = np.load(args.content_vector)
        position_vector = np.load(args.position_vector)
        
        # Reconstruct
        result = transformer.reconstruct_from_dual_vectors(
            content_vector, position_vector,
            k=args.k, similarity_threshold=args.similarity_threshold,
            content_file_path=args.content_vector
        )
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        
        output_md = getattr(result, "reconstructed_markdown", None) or getattr(result, "output", None)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_md or "# Error\n\nNo markdown could be reconstructed.")
            print(f"✅ Markdown reconstruit sauvegardé dans {args.output}")
        else:
            print("\n===== Markdown reconstruit =====\n")
            print(output_md)
            print("\n===============================\n")
        return
    
    print("❌ Erreur: Veuillez spécifier --encode-dual ou --reconstruct-dual")

if __name__ == "__main__":
    main()

# Architecture du pipeline HDC Markdown

## Introduction

Ce projet propose une solution innovante pour encoder des documents markdown en hypervecteurs à haute dimension (HDC), permettant leur compression, recherche sémantique et reconstruction sans conserver le texte original. L'objectif est de garantir que chaque document est fidèlement représenté par un vecteur unique, tout en permettant de retrouver les mots d'origine par similarité, et de reconstruire un markdown approché via un LLM.

## Problématique et solutions apportées

- **Compression sémantique** : transformer un document texte en un vecteur fixe, compact, quels que soient sa taille ou son contenu.
- **Recherche sémantique** : retrouver les mots (ou concepts) présents dans le texte à partir du vecteur uniquement, sans accès au texte original.
- **Reconstruction** : régénérer un markdown approché à partir du vecteur, grâce à un LLM et à la liste des mots les plus proches.
- **Fidélité HDC** : garantir que l'encodage respecte le paradigme HDC (somme de vecteurs mots, information distribuée, pas de triche ni de stockage caché).

## Architecture Globale

```mermaid
graph TD
    A[Markdown (.md)] -->|Prétraitement| B[Tokenisation]
    B -->|Vérification vocabulaire| C[Dictionnaire HDC (ItemMemory)]
    C -->|Vecteurs mots| D[HDC Encoder]
    D -->|Somme + Threshold| E[Hypervecteur du document (npy)]
    E -->|Recherche de similarité| F[Vector Database]
    F -->|Top-k mots similaires| G[LLM Reconstructor]
    G -->|Génération| H[Markdown reconstruit]
```

## Description des modules principaux

### 1. Prétraitement & Tokenisation
- Nettoyage du markdown, extraction des tokens (mots).
- Option : tokenisation spaCy ou simple split.

### 2. Dictionnaire HDC (ItemMemory)
- Associe chaque mot du vocabulaire à un hypervecteur unique (bipolaire, ±1, dimension configurable).
- Généré à partir d'une liste de vrais mots anglais (english_vocab.txt, WordNet, etc.).
- Stocké dans un fichier cache pour accélérer les runs.

### 3. HDC Encoder
- Encode le document comme la **somme** des vecteurs mots présents (option : majority vote/thresholding).
- Pas d'encodage positionnel ni de pondération TF-IDF (mode fidélité).
- Produit un vecteur numpy de dimension fixe (ex : 10 000).

### 4. Vector Database
- Stocke tous les vecteurs du dictionnaire HDC (un par mot).
- Permet la recherche de similarité (cosinus) entre l'hypervecteur du document et chaque mot du dico.

### 5. Recherche de similarité
- Pour un hypervecteur donné, retourne les k mots du dico les plus proches (top-k similarité).
- Ces mots sont supposés être ceux présents dans le markdown d'origine.

### 6. LLM Reconstructor
- Prend la liste des mots candidats et génère un markdown approché via un LLM (Gemini, OpenAI, etc.).
- Ne voit jamais le markdown original, seulement les mots retrouvés.

## Flux détaillé

1. **Encodage** : markdown → tokens → vecteurs mots → somme → hypervecteur (npy)
2. **Reconstruction** : hypervecteur → recherche de similarité → top-k mots → LLM → markdown reconstruit

## Garanties et diagnostics
- **Aucune triche** : pas de stockage du texte original, pas de mapping direct.
- **Logs détaillés** : couverture du dico, mots hors vocabulaire, scores de similarité, candidats transmis au LLM.
- **Tests de fidélité** : encodage d'un mot seul, comparaison doc/dico, analyse des top candidats.

## Extensions possibles
- Visualisation 3D des hypervecteurs (PCA, t-SNE, etc.).
- Adaptation à d'autres langues ou vocabulaires.
- Compression, anonymisation, clustering de documents.

---


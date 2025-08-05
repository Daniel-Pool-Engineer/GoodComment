# NLP Comment Analysis with TinyLlama and DPR

This project demonstrates a Natural Language Processing (NLP) pipeline for analyzing comments using TinyLlama for text generation and DPR (Dense Passage Retrieval) for text embeddings visualization.

## Features

- Text classification (good/bad) using TinyLlama-1.1B-Chat model
- Text parsing and preprocessing
- 3D t-SNE visualization of text embeddings
- DPR (Dense Passage Retrieval) context and question encoders
- Hugging Face model integration

## Requirements

- Python 3.7+
- Required Python packages:
  - transformers
  - torch
  - numpy
  - matplotlib
  - scikit-learn
  - huggingface_hub
  - pyasn1_modules

## Installation
1. Create a Hugging Face account and get a API_Key
2. Clone this repository:
   ```bash
   git clone [https://github.com/Daniel-Pool-Engineer/GoodComment]
   cd [GoodComment]

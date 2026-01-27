#!/bin/bash
# Install Python Dependencies
pip install -r requirements.txt;

# Install Environment Dependencies via `conda`
conda install -c pytorch faiss-cpu;
conda install -c conda-forge openjdk=21;

# Download spaCy large NLP model
python -m spacy download zh_core_web_sm

# Build search engine index
cd search_engine
mkdir -p resources resources_100 resources_1k resources_100k
python convert_product_file_format.py # convert items.json => required doc format
mkdir -p indexes
./run_indexing.sh
cd ..

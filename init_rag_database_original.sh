if [ -d "./chroma_db" ]; then
    rm -rf ./chroma_db
fi

CUDA_VISIBLE_DEVICES=1 python3 embedding.py

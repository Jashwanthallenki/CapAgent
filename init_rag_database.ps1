# Delete the chroma_db directory if it exists
if (Test-Path "./chroma_db") {
    Remove-Item -Recurse -Force "./chroma_db"
}

# Run the embedding script
$env:CUDA_VISIBLE_DEVICES = "1"
python embedding.py

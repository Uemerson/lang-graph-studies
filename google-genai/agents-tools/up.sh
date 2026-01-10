docker build -t lang-graph-studies -f Dockerfile .
docker run --rm \
    --env-file .env \
    -p 8000:8000 \
    -v "$(pwd):/app" \
    lang-graph-studies
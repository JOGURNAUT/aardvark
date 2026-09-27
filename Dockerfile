# Beaver — container image for Azure Container Apps
#
# Two things matter here, both about image size and cold start:
#
#  1. torch is NOT in requirements.txt — sentence-transformers pulls it in
#     transitively, and the default PyPI wheel is the CUDA build (~4 GB).
#     Installing the CPU-only wheel FIRST means pip sees torch already
#     satisfied later, and the image lands around 1 GB instead.
#
#  2. all-MiniLM-L6-v2 (~80 MB) is downloaded on first use. Baking it into
#     the image at build time keeps cold starts fast and makes the container
#     work even if Hugging Face is unreachable.

# Stage 1: the React front end.
#
# Node is needed to build it and not to serve it, so it stays out of the final
# image entirely. Only web/dist crosses over, which is ~150 KB of JS and CSS
# against the ~700 MB of Node and its module tree that would otherwise ship.
FROM node:20-slim AS web
WORKDIR /web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build


# Stage 2: the application image.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# CPU-only torch, before anything can drag in the CUDA build
RUN pip install --no-cache-dir torch \
      --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

COPY . .

# The built assets. api/server.py mounts this directory when it exists.
COPY --from=web /web/dist ./web/dist

# Bake the embedding model into the image
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

# SQLite lives here. Mount an Azure Files share at /app/data to make it
# survive restarts (see deploy-azure.sh, step 5).
ENV DB_PATH=/app/data/research.db
RUN mkdir -p /app/data

EXPOSE 8000

# The FastAPI app serves both the JSON API and the built React assets, so one
# process and one port covers the whole front end.
#
# The Streamlit UI is still in the image and still works:
#   docker run -p 8501:8501 beaver streamlit run ui/app.py
#     --server.port=8501 --server.address=0.0.0.0
#
# One worker on purpose. The agent holds the embedding model in process, so a
# second worker would load a second copy of it for no throughput gain at this
# scale, and Container Apps already runs a single replica because the UI keeps
# session state in the server process.
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

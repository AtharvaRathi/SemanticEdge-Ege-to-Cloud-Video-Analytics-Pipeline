# SemanticEdge Frontend

Minimalist React client for natural-language zero-shot visual video search.

## Features
- **Natural Language Search Bar**: Query edge-captured keyframes using arbitrary text prompts (e.g., `"person in black hoodie running"`, `"white delivery van stopping"`).
- **FastAPI & Qdrant Connection**: Connects to the cloud search endpoint (`POST /api/v1/search`) with sub-10ms ANN vector retrieval.
- **Rich Metadata Inspection**: Renders similarity score matches, frame indices, capture timestamps, object counts, and tracking IDs.
- **Dark Theme UI**: Clean, responsive layout with status telemetry and query preset chips.

## Local Development
```bash
# Install dependencies
npm install

# Start Vite dev server on http://localhost:3000
npm run dev
```

## Production Build
```bash
# Compile optimized production bundle to /dist
npm run build

# Preview build locally
npm run preview
```

## Docker Deployment
Handled automatically via the root `docker-compose.yml`:
```bash
docker compose up -d frontend
```
The application will be served via Nginx on `http://localhost:3000`.

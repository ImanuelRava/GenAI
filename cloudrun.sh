#!/usr/bin/env bash
# ==========================================================================
# Google Cloud Run — build & deploy
# Usage: bash deploy/cloudrun.sh <IMAGE_NAME> <REGION>
# Example: bash deploy/cloudrun.sh genai-platform us-central1
# ==========================================================================
set -euo pipefail

IMAGE="${1:-genai-research-platform}"
REGION="${2:-us-central1}"
TAG="gcr.io/$(gcloud config get-value project)/${IMAGE}"

# Build
printf "Building image: %s ...\n" "$TAG"
docker build -t "$TAG" .

# Push
docker push "$TAG"

# Deploy
gcloud run deploy "$IMAGE" \
  --image "$TAG" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 120 \
  --set-env-vars "GUNICORN_WORKERS=2" \
  --set-env-vars "GUNICORN_TIMEOUT=120"

printf "Deployed to: %s\n" "$(gcloud run services describe "$IMAGE" --region "$REGION" --format 'value(status.url)')"

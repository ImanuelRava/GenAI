#!/usr/bin/env bash
# ==========================================================================
# Google Cloud Run — build & deploy
# Usage:
#   bash deploy/cloudrun.sh <PROJECT_ID> [REGION]
# Example:
#   bash deploy/cloudrun.sh genai-platform-imanuel us-central1
# ==========================================================================
set -euo pipefail

PROJECT_ID="${1:?Usage: bash deploy/cloudrun.sh <PROJECT_ID> [REGION]}"
REGION="${2:-us-central1}"
IMAGE="gcr.io/${PROJECT_ID}/genai-platform"
SERVICE_NAME="genai-platform"

# Step 1: Set project
echo "==> Setting project to ${PROJECT_ID}"
gcloud config set project "$PROJECT_ID"

# Step 2: Enable APIs (safe to run multiple times)
echo "==> Enabling required APIs..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  containerregistry.googleapis.com \
  --project "$PROJECT_ID"

# Step 3: Authenticate Docker
echo "==> Configuring Docker auth..."
gcloud auth configure-docker --quiet

# Step 4: Build & Deploy (one step)
echo "==> Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 120 \
  --project "$PROJECT_ID"

# Step 5: Print the URL
URL=$(gcloud run services describe "$SERVICE_NAME" \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --format 'value(status.url)')

echo ""
echo "=========================================="
echo "  Deployed successfully!"
echo "  URL: ${URL}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Open the URL above in your browser"
echo "  2. Set CORS (replace URL below):"
echo "     gcloud run services update ${SERVICE_NAME} \\
       --region ${REGION} \\
       --update-env-vars \"CORS_ORIGINS=${URL}\""
echo ""

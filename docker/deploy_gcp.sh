#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy_gcp.sh — Deploy ClinicalNER to GCP Cloud Run (free tier)
# Free tier: 2M requests/month, 360k GB-seconds compute/month
#
# Prerequisites:
#   1. gcloud CLI installed and authenticated (gcloud auth login)
#   2. Docker installed locally
#   3. GCP project created
#
# Usage:
#   chmod +x docker/deploy_gcp.sh
#   ./docker/deploy_gcp.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

GCP_PROJECT="your-gcp-project-id"    # update this
REGION="us-central1"
SERVICE_NAME="clinicalner"
IMAGE="gcr.io/${GCP_PROJECT}/${SERVICE_NAME}"

echo ""
echo "════════════════════════════════════════════════════"
echo "  ClinicalNER — GCP Cloud Run Deployment"
echo "════════════════════════════════════════════════════"

echo "► Building and pushing to Google Container Registry..."
gcloud builds submit \
    --tag ${IMAGE} \
    --project ${GCP_PROJECT} \
    .

echo "► Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --port 5000 \
    --memory 512Mi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 3 \
    --project ${GCP_PROJECT}

SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --region ${REGION} \
    --format 'value(status.url)' \
    --project ${GCP_PROJECT})

echo ""
echo "════════════════════════════════════════════════════"
echo "  Deployment complete!"
echo "────────────────────────────────────────────────────"
echo "  Service URL : ${SERVICE_URL}"
echo "  Dashboard   : ${SERVICE_URL}/dashboard"
echo "  API         : ${SERVICE_URL}/api/deidentify"
echo "  Health      : ${SERVICE_URL}/health"
echo "════════════════════════════════════════════════════"
echo ""

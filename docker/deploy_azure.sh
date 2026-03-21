#!/bin/bash
# -----------------------------------------------------------------------------
# deploy_azure.sh - Deploy ClinicalNER to Azure App Service (Linux Container)
#
# Prerequisites:
#   - Azure CLI installed
#   - az login completed
#   - Execute from repository root
#
# Usage:
#   chmod +x docker/deploy_azure.sh
#   AZURE_WEBAPP_NAME=clinicalner-demo-12345 AZURE_ACR_NAME=clinicalneracr12345 ./docker/deploy_azure.sh
# -----------------------------------------------------------------------------

set -euo pipefail

AZURE_RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-clinicalner-rg}"
AZURE_LOCATION="${AZURE_LOCATION:-eastus}"
AZURE_PLAN_NAME="${AZURE_PLAN_NAME:-clinicalner-plan}"
AZURE_WEBAPP_NAME="${AZURE_WEBAPP_NAME:-}"
AZURE_ACR_NAME="${AZURE_ACR_NAME:-}"
AZURE_IMAGE_NAME="${AZURE_IMAGE_NAME:-clinicalner}"
AZURE_IMAGE_TAG="${AZURE_IMAGE_TAG:-latest}"

if [[ -z "${AZURE_WEBAPP_NAME}" ]]; then
  echo "ERROR: AZURE_WEBAPP_NAME is required and must be globally unique."
  echo "Example: AZURE_WEBAPP_NAME=clinicalner-demo-12345"
  exit 1
fi

if [[ -z "${AZURE_ACR_NAME}" ]]; then
  echo "ERROR: AZURE_ACR_NAME is required and must be globally unique."
  echo "Example: AZURE_ACR_NAME=clinicalneracr12345"
  exit 1
fi

if ! command -v az >/dev/null 2>&1; then
  echo "ERROR: Azure CLI is not installed or not in PATH."
  exit 1
fi

IMAGE_URI="${AZURE_ACR_NAME}.azurecr.io/${AZURE_IMAGE_NAME}:${AZURE_IMAGE_TAG}"

echo ""
echo "============================================================"
echo " ClinicalNER - Azure App Service Deployment"
echo "============================================================"
echo "Resource Group : ${AZURE_RESOURCE_GROUP}"
echo "Location       : ${AZURE_LOCATION}"
echo "Plan           : ${AZURE_PLAN_NAME}"
echo "Web App        : ${AZURE_WEBAPP_NAME}"
echo "ACR            : ${AZURE_ACR_NAME}"
echo "Image          : ${IMAGE_URI}"

# 1) Resource group
az group create --name "${AZURE_RESOURCE_GROUP}" --location "${AZURE_LOCATION}" >/dev/null

# 2) ACR (create if missing)
if ! az acr show --name "${AZURE_ACR_NAME}" --resource-group "${AZURE_RESOURCE_GROUP}" >/dev/null 2>&1; then
  az acr create \
    --resource-group "${AZURE_RESOURCE_GROUP}" \
    --name "${AZURE_ACR_NAME}" \
    --sku Basic \
    --admin-enabled true >/dev/null
fi

# 3) Build image in ACR
az acr build \
  --registry "${AZURE_ACR_NAME}" \
  --image "${AZURE_IMAGE_NAME}:${AZURE_IMAGE_TAG}" \
  --file docker/Dockerfile \
  .

# 4) App Service plan (create if missing)
if ! az appservice plan show --name "${AZURE_PLAN_NAME}" --resource-group "${AZURE_RESOURCE_GROUP}" >/dev/null 2>&1; then
  az appservice plan create \
    --name "${AZURE_PLAN_NAME}" \
    --resource-group "${AZURE_RESOURCE_GROUP}" \
    --is-linux \
    --sku B1 >/dev/null
fi

# 5) Web app (create if missing)
if ! az webapp show --name "${AZURE_WEBAPP_NAME}" --resource-group "${AZURE_RESOURCE_GROUP}" >/dev/null 2>&1; then
  az webapp create \
    --resource-group "${AZURE_RESOURCE_GROUP}" \
    --plan "${AZURE_PLAN_NAME}" \
    --name "${AZURE_WEBAPP_NAME}" \
    --container-image-name "${IMAGE_URI}" >/dev/null
fi

# 6) Registry credentials
ACR_USER=$(az acr credential show --name "${AZURE_ACR_NAME}" --query username -o tsv)
ACR_PASS=$(az acr credential show --name "${AZURE_ACR_NAME}" --query passwords[0].value -o tsv)

az webapp config container set \
  --resource-group "${AZURE_RESOURCE_GROUP}" \
  --name "${AZURE_WEBAPP_NAME}" \
  --container-image-name "${IMAGE_URI}" \
  --container-registry-url "https://${AZURE_ACR_NAME}.azurecr.io" \
  --container-registry-user "${ACR_USER}" \
  --container-registry-password "${ACR_PASS}" >/dev/null

# 7) App settings
az webapp config appsettings set \
  --resource-group "${AZURE_RESOURCE_GROUP}" \
  --name "${AZURE_WEBAPP_NAME}" \
  --settings \
    FLASK_ENV=production \
    WEBSITES_PORT=5000 \
    DB_PATH=/home/clinicalner.db >/dev/null

APP_URL="https://${AZURE_WEBAPP_NAME}.azurewebsites.net"

echo ""
echo "============================================================"
echo " Deployment complete"
echo "------------------------------------------------------------"
echo " App URL   : ${APP_URL}"
echo " Health    : ${APP_URL}/health"
echo " Dashboard : ${APP_URL}/dashboard"
echo " API       : ${APP_URL}/api/deidentify"
echo ""
echo "Next step: In Azure Portal set Health check path to /health"
echo "============================================================"

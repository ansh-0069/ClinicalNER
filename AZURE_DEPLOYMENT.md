# Azure Deployment (App Service + ACR)

This guide deploys ClinicalNER as a Linux container on Azure App Service using Azure Container Registry (ACR).

## Prerequisites

- Azure subscription
- Azure CLI installed and logged in (`az login`)
- Bash shell (Git Bash / WSL on Windows)

## Quick Deploy (Script)

Use the deployment script in [docker/deploy_azure.sh](docker/deploy_azure.sh).

```bash
chmod +x docker/deploy_azure.sh

# Required: set a globally unique app name
AZURE_WEBAPP_NAME=clinicalner-demo-12345 \
AZURE_ACR_NAME=clinicalneracr12345 \
./docker/deploy_azure.sh
```

Optional environment variables:

- `AZURE_LOCATION` (default: `eastus`)
- `AZURE_RESOURCE_GROUP` (default: `clinicalner-rg`)
- `AZURE_PLAN_NAME` (default: `clinicalner-plan`)
- `AZURE_IMAGE_NAME` (default: `clinicalner`)
- `AZURE_IMAGE_TAG` (default: `latest`)

## Manual Deploy Steps

1. Create resource group:

```bash
az group create --name clinicalner-rg --location eastus
```

2. Create ACR:

```bash
az acr create --resource-group clinicalner-rg --name clinicalneracr12345 --sku Basic --admin-enabled true
```

3. Build image in ACR:

```bash
az acr build --registry clinicalneracr12345 --image clinicalner:latest --file docker/Dockerfile .
```

4. Create Linux App Service plan:

```bash
az appservice plan create --name clinicalner-plan --resource-group clinicalner-rg --is-linux --sku B1
```

5. Create Web App with container image:

```bash
az webapp create \
  --resource-group clinicalner-rg \
  --plan clinicalner-plan \
  --name clinicalner-demo-12345 \
  --container-image-name clinicalneracr12345.azurecr.io/clinicalner:latest
```

6. Configure registry credentials and app settings:

```bash
ACR_USER=$(az acr credential show --name clinicalneracr12345 --query username -o tsv)
ACR_PASS=$(az acr credential show --name clinicalneracr12345 --query passwords[0].value -o tsv)

az webapp config container set \
  --resource-group clinicalner-rg \
  --name clinicalner-demo-12345 \
  --container-image-name clinicalneracr12345.azurecr.io/clinicalner:latest \
  --container-registry-url https://clinicalneracr12345.azurecr.io \
  --container-registry-user "$ACR_USER" \
  --container-registry-password "$ACR_PASS"

az webapp config appsettings set \
  --resource-group clinicalner-rg \
  --name clinicalner-demo-12345 \
  --settings FLASK_ENV=production WEBSITES_PORT=5000 DB_PATH=/home/clinicalner.db
```

7. Configure health check path (`/health`) in Azure Portal:

- App Service -> Settings -> Health check -> Path: `/health`

8. Validate:

- `/health`
- `/`
- `/dashboard`
- `/api/stats`

## Notes

- For demo, SQLite is fine (`DB_PATH=/home/clinicalner.db`).
- App Service storage can reset depending on configuration; persistent database is recommended for production.

#!/usr/bin/env bash
# Deploy Aardvark to Azure Container Apps.
#
# Prereq: Azure CLI. If `az` is not found:
#     winget install -e --id Microsoft.AzureCLI
#     (then reopen the terminal)
#
# Run:  bash deploy-azure.sh
set -euo pipefail

# These are the LIVE resource names and they still say beaver. Azure
# Container Apps cannot be renamed in place: changing APP creates a second
# app on a new hostname and abandons the current one, which is a URL on a
# CV. Renaming the product did not rename the infrastructure, and pretending
# otherwise here would break the script against what is actually deployed.
RG="beaver-rg"
APP="beaver"
LOCATION="centralindia"

# A terminal that was already open when the CLI was installed still has the old
# PATH, so fall back to the default install location before giving up.
if ! command -v az >/dev/null 2>&1; then
  for candidate in \
    "/c/Program Files/Microsoft SDKs/Azure/CLI2/wbin" \
    "/c/Program Files (x86)/Microsoft SDKs/Azure/CLI2/wbin"; do
    if [[ -f "$candidate/az.cmd" ]]; then PATH="$candidate:$PATH"; break; fi
  done
fi
if ! command -v az >/dev/null 2>&1; then
  echo "!! az not found. Install it with:"
  echo "     winget install -e --id Microsoft.AzureCLI"
  echo "   then open a new terminal."
  exit 1
fi

# ---------------------------------------------------------------- 1. keys
# Read from your local .env so nothing is typed into the shell history.
if [[ -f .env ]]; then
  set -a; source .env; set +a
else
  echo "!! .env not found. Copy .env.example to .env and fill in the keys."
  exit 1
fi

for key in GROQ_API_KEY GEMINI_API_KEY TAVILY_API_KEY; do
  if [[ -z "${!key:-}" ]]; then echo "!! $key is empty in .env"; exit 1; fi
done

# ---------------------------------------------------------------- 2. login
if az account show >/dev/null 2>&1; then
  echo "Already signed in as $(az account show --query user.name -o tsv)"
else
  az login
fi
az extension add --name containerapp --upgrade --only-show-errors
# A fresh subscription has none of these registered. ContainerRegistry is the
# one that bites: `containerapp up --source .` builds through ACR, so without
# it the deploy dies partway with MissingSubscriptionRegistration.
for ns in Microsoft.App Microsoft.OperationalInsights Microsoft.ContainerRegistry Microsoft.Storage; do
  echo "registering $ns ..."
  az provider register --namespace "$ns" --wait
done

# ---------------------------------------------------------------- 3. deploy
# `--source .` builds the image in Azure (ACR Tasks) — no local Docker push,
# no registry to set up by hand.
az group create -n "$RG" -l "$LOCATION" -o none

az containerapp up \
  --name "$APP" \
  --resource-group "$RG" \
  --location "$LOCATION" \
  --source . \
  --ingress external \
  --target-port 8501

# ---------------------------------------------------------------- 4. secrets
# API keys as Container Apps secrets, referenced by env vars. They are never
# baked into the image and never show up in `az containerapp show`.
az containerapp secret set -n "$APP" -g "$RG" --only-show-errors \
  --secrets groq="$GROQ_API_KEY" \
            gemini="$GEMINI_API_KEY" \
            tavily="$TAVILY_API_KEY"

# max-replicas 1: Streamlit keeps session state in the server process, so a
#                 second replica would silently break sessions.
# min-replicas 0: scales to zero when idle => costs nothing when nobody is on it.
#                 Trade-off is a slow first request while the model loads.
# MSYS_NO_PATHCONV: Git Bash rewrites anything that looks like a Unix path into
# a Windows one, so DB_PATH=/app/data/research.db would reach Azure as
# "C:/Program Files/Git/app/data/research.db". Harmless-looking, and the app
# still starts — it just writes the database to the wrong place.
MSYS_NO_PATHCONV=1 \
az containerapp update -n "$APP" -g "$RG" --only-show-errors \
  --set-env-vars GROQ_API_KEY=secretref:groq \
                 GEMINI_API_KEY=secretref:gemini \
                 TAVILY_API_KEY=secretref:tavily \
                 DB_PATH=/app/data/research.db \
  --cpu 1 --memory 2Gi \
  --min-replicas 0 --max-replicas 1

echo
echo "Live at: https://$(az containerapp show -n "$APP" -g "$RG" \
  --query properties.configuration.ingress.fqdn -o tsv)"
echo
echo "Logs:    az containerapp logs show -n $APP -g $RG --follow"
echo "Delete:  az group delete -n $RG --yes --no-wait"

# ------------------------------------------------- 5. OPTIONAL: persist SQLite
# Without this the database resets whenever the container restarts — which,
# with min-replicas 0, is every time it scales back up. Fine for a demo;
# uncomment to keep chat history across restarts.
#
# STORAGE="beaverstore$RANDOM"
# az storage account create -n "$STORAGE" -g "$RG" -l "$LOCATION" --sku Standard_LRS -o none
# KEY=$(az storage account keys list -n "$STORAGE" -g "$RG" --query "[0].value" -o tsv)
# az storage share create -n beaverdata --account-name "$STORAGE" --account-key "$KEY" -o none
#
# ENV_NAME=$(az containerapp show -n "$APP" -g "$RG" --query properties.environmentId -o tsv | sed 's|.*/||')
# az containerapp env storage set -n "$ENV_NAME" -g "$RG" \
#   --storage-name beaverfiles --azure-file-account-name "$STORAGE" \
#   --azure-file-account-key "$KEY" --azure-file-share-name beaverdata \
#   --access-mode ReadWrite -o none
#
# Then add the volume to the app template (needs a YAML patch):
#   az containerapp show -n "$APP" -g "$RG" -o yaml > app.yaml
#   # under template:, add:
#   #   volumes:
#   #     - name: data
#   #       storageName: beaverfiles
#   #       storageType: AzureFile
#   #   containers[0].volumeMounts:
#   #     - volumeName: data
#   #       mountPath: /app/data
#   az containerapp update -n "$APP" -g "$RG" --yaml app.yaml

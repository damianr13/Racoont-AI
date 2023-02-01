# Racoont AI
Generate bed time stories with Racoont AI.



# Deployment on GCP

Trigger the build on GCP
```bash
gcloud builds submit --tag gcr.io/news-bots-342522/racoont-ai . 
```

Deploy the app on GCP with the following command:

```bash
gcloud run deploy racoont-ai --image gcr.io/news-bots-342522/racoont-ai \
    --execution-environment gen2 \
    --allow-unauthenticated \
    --service-account fs-identity \
    --update-env-vars BUCKET=racoont-data,MNT_DIR=/mnt/gcs,COHERE_API_KEY=$API_KEY \
    --region europe-west1
    --memory 3Gi
```
# Racoont AI
Generate bed time stories with Racoont AI.

Based on the Cohere API (https://www.cohere.ai/), we've build a tool that is able to: 
- generate a story with a given morale that the child should learn
- create the main character of the story based on the input 
- read out the story in a custom voice generated from only 1 minute of audio

# How to use it

The app runs as a streamlit app at the following url: 
https://racoont-ai-xu2xz4qutq-ew.a.run.app/

# Details about the implementation

- Based on the input we first look through our database of fables
- In our database each fable has its morale associated with it
- We use cohere embed API to find the fable with the closest morale to the input
- The identified fable and the name of the character are used to create a prompt for the story generation
- For each prompt we generate 5 stories and select the one with the highest score [SPACHE](https://readable.com/readability/spache-readability-formula/) score
- Finally, we use a voice sample of 1 minute to generate a custom voice for the story with [Coqui AI TTS](https://github.com/coqui-ai/TTS)

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
    --region europe-west1 \
    --update-env-vars BUCKET=racoont-data,MNT_DIR=/mnt/gcs,COHERE_API_KEY=$API_KEY 
```
FROM python:3.10

WORKDIR /app
RUN apt-get update && apt-get install -y \
    libasound2-dev \
    libsndfile1-dev \
    ffmpeg \
    tini \
    lsb-release; \
    gcsFuseRepo=gcsfuse-`lsb_release -c -s`; \
    echo "deb http://packages.cloud.google.com/apt $gcsFuseRepo main" | \
    tee /etc/apt/sources.list.d/gcsfuse.list; \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    apt-key add -; \
    apt-get update; \
    apt-get install -y gcsfuse \
    && rm -rf /var/lib/apt/lists/*

COPY . ./
RUN pip install -r requirements.txt

# Ensure the script is executable
RUN chmod +x /app/run.sh

# Use tini to manage zombie processes and signal forwarding
# https://github.com/krallin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/run.sh"]
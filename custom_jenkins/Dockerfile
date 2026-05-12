# ─────────────────────────────────────────────────────────────────────────────
# custom_jenkins/Dockerfile
# Custom Jenkins image with Python 3, pip, and Google Cloud CLI pre-installed.
#
# Build:
#   docker build -t custom-jenkins ./custom_jenkins
#
# Run:
#   docker run -d \
#     --name jenkins \
#     -p 8080:8080 \
#     -p 50000:50000 \
#     -v jenkins_home:/var/jenkins_home \
#     -v /var/run/docker.sock:/var/run/docker.sock \
#     custom-jenkins
# ─────────────────────────────────────────────────────────────────────────────

FROM jenkins/jenkins:lts

USER root

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        libgomp1 \
        curl \
        apt-transport-https \
        ca-certificates \
        gnupg \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# libgomp1 is required by LightGBM (OpenMP)

# ── Google Cloud CLI ──────────────────────────────────────────────────────────
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
        gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] \
        https://packages.cloud.google.com/apt cloud-sdk main" | \
        tee /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update -y && \
    apt-get install -y google-cloud-cli && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ── Docker CLI ───────────────────────────────────────────────────────────────
# Install the Docker CLI so Jenkins can run `docker build`, `docker push`, etc.
# The actual Docker daemon comes from the host via socket mount (no DinD needed)
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | \
        gpg --dearmor -o /usr/share/keyrings/docker.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker.gpg] \
        https://download.docker.com/linux/debian bookworm stable" | \
        tee /etc/apt/sources.list.d/docker.list && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends docker-ce-cli && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ── Docker socket access ──────────────────────────────────────────────────────
# Give jenkins user access to the host Docker socket
RUN groupadd -f docker && \
    usermod -aG docker jenkins

# ── Switch back to Jenkins user ───────────────────────────────────────────────
USER jenkins

# ── Verify installations ──────────────────────────────────────────────────────
RUN python3 --version && \
    gcloud --version

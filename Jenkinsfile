// ─────────────────────────────────────────────────────────────────────────────
// Jenkinsfile — Loan Credit Default Risk MLOps Pipeline
//
// Stages:
//   1. Checkout          — clone the repo
//   2. Install Deps      — set up Python venv + install requirements
//   3. Pull Artifacts    — download model pkl files from GCP
//   4. Run Tests         — pytest with coverage (all branches)
//   5. Retrain Model     — only on main branch
//   6. Drift Monitoring  — only on main branch
//   7. Push Artifacts    — only on main branch
//   8. Build Image       — docker build (only on main branch)
//   9. Push to GCR       — push image to GCP Artifact Registry (only on main)
//  10. Deploy            — Cloud Run deploy (only on main branch)
// ─────────────────────────────────────────────────────────────────────────────

pipeline {
    agent any

    // ── Pipeline-wide environment ─────────────────────────────────────────────
    // environment {
    //     // GCP settings — update PROJECT_ID and REGION if needed
    //     GCP_PROJECT_ID          = 'mlops-495517'
    //     GCP_REGION              = 'us-central1'
    //     GCP_BUCKET              = 'mlops_b1'
    //     GAR_HOSTNAME            = 'us-central1-docker.pkg.dev'
    //     GAR_REPO                = 'loan-default-repo'
    //     IMAGE_NAME              = 'loan-default-api'
    //     CLOUD_RUN_SERVICE       = 'loan-default-api'

    //     // Full image URI (tag set per build)
    //     IMAGE_URI = "${GAR_HOSTNAME}/${GCP_PROJECT_ID}/${GAR_REPO}/${IMAGE_NAME}:${BUILD_NUMBER}"
    //     IMAGE_URI_LATEST = "${GAR_HOSTNAME}/${GCP_PROJECT_ID}/${GAR_REPO}/${IMAGE_NAME}:latest"

    //     // GCP service account key — stored in Jenkins credentials store
    //     GOOGLE_APPLICATION_CREDENTIALS = credentials('gcp-service-account-key')

    //     // Python venv inside the workspace
    //     VENV_DIR = "${WORKSPACE}/venv"
    // }

    options {
        // Keep the last 10 builds
        buildDiscarder(logRotator(numToKeepStr: '10'))
        // Abort if the pipeline hasn't finished in 60 minutes
        timeout(time: 60, unit: 'MINUTES')
        // Don't run concurrent builds on the same branch
        disableConcurrentBuilds()
        // Add timestamps to console log
        timestamps()
    }

    stages {

        // ── Stage 1: Checkout ─────────────────────────────────────────────────
        stage('Checkout') {
    steps {
        echo "Branch: ${env.BRANCH_NAME ?: 'unknown'}"
        checkout scmGit(
            branches: [[name: '*/main']],
            extensions: [],
            userRemoteConfigs: [[
                credentialsId: 'github',
                url: 'https://github.com/salmazoya/Loan_credit_default_MLOPS.git']]
                )
            }
        }

    //     // ── Stage 2: Install Python Dependencies ─────────────────────────────
    //     stage('Install Dependencies') {
    //         steps {
    //             sh '''
    //                 set -e
    //                 echo "==> Creating Python virtual environment..."
    //                 python3 -m venv ${VENV_DIR}
    //                 . ${VENV_DIR}/bin/activate

    //                 echo "==> Upgrading pip..."
    //                 pip install --upgrade pip --quiet

    //                 echo "==> Installing project requirements..."
    //                 pip install -r requirements.txt --quiet

    //                 echo "==> Installing test dependencies..."
    //                 pip install pytest pytest-cov httpx --quiet

    //                 echo "==> Python version:"
    //                 python --version

    //                 echo "==> Key packages:"
    //                 pip show lightgbm | grep Version
    //                 pip show fastapi   | grep Version
    //                 pip show evidently | grep Version
    //             '''
    //         }
    //     }

    //     // ── Stage 3: Pull Model Artifacts from GCP ────────────────────────────
    //     stage('Pull Artifacts from GCP') {
    //         steps {
    //             sh '''
    //                 set -e
    //                 . ${VENV_DIR}/bin/activate

    //                 echo "==> Authenticating with GCP..."
    //                 gcloud auth activate-service-account \
    //                     --key-file="${GOOGLE_APPLICATION_CREDENTIALS}" \
    //                     --quiet
    //                 gcloud config set project ${GCP_PROJECT_ID} --quiet

    //                 echo "==> Pulling model artifacts from GCS bucket: ${GCP_BUCKET}..."
    //                 python -m src.gcp_sync --pull --group models

    //                 echo "==> Artifacts pulled:"
    //                 ls -lh artifacts/models/ || echo "No artifacts/models directory found"
    //             '''
    //         }
    //     }

    //     // ── Stage 4: Run Tests ────────────────────────────────────────────────
    //     stage('Run Tests') {
    //         steps {
    //             sh '''
    //                 set -e
    //                 . ${VENV_DIR}/bin/activate

    //                 echo "==> Running pytest with coverage..."
    //                 pytest tests/ \
    //                     -v \
    //                     --cov=src \
    //                     --cov=main \
    //                     --cov-report=xml:coverage.xml \
    //                     --cov-report=term-missing \
    //                     --junitxml=test-results.xml \
    //                     --tb=short
    //             '''
    //         }
    //         post {
    //             always {
    //                 // Publish JUnit test results
    //                 junit 'test-results.xml'
    //             }
    //         }
    //     }

    //     // ── Stage 5: Retrain Model (main only) ────────────────────────────────
    //     stage('Retrain Model') {
    //         when {
    //             branch 'main'
    //         }
    //         steps {
    //             sh '''
    //                 set -e
    //                 . ${VENV_DIR}/bin/activate

    //                 echo "==> Pulling raw + processed data from GCP for retraining..."
    //                 python -m src.gcp_sync --pull --group raw
    //                 python -m src.gcp_sync --pull --group processed

    //                 echo "==> Running training pipeline..."
    //                 python -m src.train

    //                 echo "==> Training complete."
    //             '''
    //         }
    //     }

    //     // ── Stage 6: Drift Monitoring (main only) ─────────────────────────────
    //     stage('Drift Monitoring') {
    //         when {
    //             branch 'main'
    //         }
    //         steps {
    //             sh '''
    //                 set -e
    //                 . ${VENV_DIR}/bin/activate

    //                 echo "==> Running Evidently AI drift monitoring..."
    //                 python -m src.monitor

    //                 echo "==> Drift reports saved to artifacts/monitoring/"
    //                 ls -lh artifacts/monitoring/ || true
    //             '''
    //         }
    //     }

    //     // ── Stage 7: Push Artifacts to GCP (main only) ───────────────────────
    //     stage('Push Artifacts to GCP') {
    //         when {
    //             branch 'main'
    //         }
    //         steps {
    //             sh '''
    //                 set -e
    //                 . ${VENV_DIR}/bin/activate

    //                 echo "==> Pushing model artifacts to GCS bucket: ${GCP_BUCKET}..."
    //                 python -m src.gcp_sync --push --group models

    //                 echo "==> Push complete."
    //             '''
    //         }
    //     }

    //     // ── Stage 8: Build Docker Image (main only) ───────────────────────────
    //     stage('Build Docker Image') {
    //         when {
    //             branch 'main'
    //         }
    //         steps {
    //             sh '''
    //                 set -e
    //                 echo "==> Building Docker image: ${IMAGE_URI}..."
    //                 docker build \
    //                     --tag ${IMAGE_URI} \
    //                     --tag ${IMAGE_URI_LATEST} \
    //                     --label "git-commit=${GIT_COMMIT}" \
    //                     --label "build-number=${BUILD_NUMBER}" \
    //                     .
    //                 echo "==> Image built successfully."
    //                 docker image ls | grep ${IMAGE_NAME}
    //             '''
    //         }
    //     }

    //     // ── Stage 9: Push Image to GCP Artifact Registry (main only) ─────────
    //     stage('Push Image to GCP Artifact Registry') {
    //         when {
    //             branch 'main'
    //         }
    //         steps {
    //             sh '''
    //                 set -e
    //                 echo "==> Configuring Docker to authenticate with GCP Artifact Registry..."
    //                 gcloud auth configure-docker ${GAR_HOSTNAME} --quiet

    //                 echo "==> Pushing image: ${IMAGE_URI}..."
    //                 docker push ${IMAGE_URI}

    //                 echo "==> Pushing latest tag..."
    //                 docker push ${IMAGE_URI_LATEST}

    //                 echo "==> Image pushed successfully."
    //             '''
    //         }
    //     }

    //     // ── Stage 10: Deploy to Cloud Run (main only) ─────────────────────────
    //     stage('Deploy to Cloud Run') {
    //         when {
    //             branch 'main'
    //         }
    //         steps {
    //             sh '''
    //                 set -e
    //                 echo "==> Deploying ${IMAGE_URI} to Cloud Run service: ${CLOUD_RUN_SERVICE}..."

    //                 gcloud run deploy ${CLOUD_RUN_SERVICE} \
    //                     --image=${IMAGE_URI} \
    //                     --platform=managed \
    //                     --region=${GCP_REGION} \
    //                     --allow-unauthenticated \
    //                     --memory=2Gi \
    //                     --cpu=2 \
    //                     --min-instances=0 \
    //                     --max-instances=3 \
    //                     --port=8080 \
    //                     --set-env-vars="GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/gcp_key.json" \
    //                     --quiet

    //                 echo "==> Deployment complete. Service URL:"
    //                 gcloud run services describe ${CLOUD_RUN_SERVICE} \
    //                     --platform=managed \
    //                     --region=${GCP_REGION} \
    //                     --format="value(status.url)"
    //             '''
    //         }
    //     }

    // } // end stages

    // // ── Post-pipeline actions ─────────────────────────────────────────────────
    // post {
    //     always {
    //         echo "==> Pipeline finished. Branch: ${env.BRANCH_NAME ?: 'unknown'}, Build: ${BUILD_NUMBER}"
    //     }
    //     success {
    //         echo "✅ Pipeline succeeded."
    //     }
    //     failure {
    //         echo "❌ Pipeline failed. Check logs above for details."
    //     }
    //     cleanup {
    //         // Remove dangling Docker images to free up disk space
    //         sh '''
    //             docker image prune -f || true
    //         '''
    //     }
    }

} // end pipeline

# #!/bin/bash

# # GCP project and repository configuration
# PROJECT_ID="your-gcp-project-id"  # Replace with your actual project ID if different
# REGION="asia-southeast1"
# REPOSITORY="m2-validation-app-repo"
# IMAGE_NAME="m2-validation-app"

# echo "Building Docker image..."
# docker build --no-cache -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME .

# echo "Pushing Docker image to Artifact Registry..."
# docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME

# echo "Deploying to Cloud Run..."
# gcloud run deploy $IMAGE_NAME \
#   --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME \
#   --platform managed \
#   --region $REGION \
#   --allow-unauthenticated


#!/bin/bash

# GCP project and repository configuration
# PROJECT_ID="your-gcp-project-id"  # Replace with your actual project ID if different
# REGION="asia-southeast1"
# REPOSITORY="m2-validation-app-repo"
# IMAGE_NAME="m2-validation-app"

# echo "Building Docker image..."
# docker build --no-cache -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME .

# echo "Pushing Docker image to Artifact Registry..."
# docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME

# echo "Deploying to Cloud Run with more resources..."
# gcloud run deploy $IMAGE_NAME \
#   --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME \
#     --platform managed \
#       --region $REGION \
#         --allow-unauthenticated \
#           --cpu=2 \
#             --memory=2Gi \
#               --concurrency=10 \
#                 --max-instances=20
                
PROJECT_ID="your-gcp-project-id"  # Replace with your actual project ID if different
REGION="asia-southeast1"
REPOSITORY="m2-validation-app-repo"
IMAGE_NAME="m2-validation-app"
echo "Building Docker image..."
docker build --no-cache -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME .
echo "Pushing Docker image to Artifact Registry..."
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME
echo "Deploying to Cloud Run with more resources..."
gcloud run deploy $IMAGE_NAME \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --no-allow-unauthenticated \
  --cpu=2 \
  --memory=2Gi \
  --concurrency=10 \
  --max-instances=20 \
  --vpc-egress=private-ranges-only \
  --network=your-vpc-network \
  --subnet=your-subnet
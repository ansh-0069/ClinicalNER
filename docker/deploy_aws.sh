#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy_aws.sh — Deploy ClinicalNER to AWS EC2 (free tier)
#
# Prerequisites:
#   1. AWS CLI installed and configured (aws configure)
#   2. Docker installed locally
#   3. An EC2 key pair created in AWS console
#
# Usage:
#   chmod +x docker/deploy_aws.sh
#   ./docker/deploy_aws.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e   # exit immediately on any error

# ── Config — update these ─────────────────────────────────────────────────────
DOCKERHUB_USER="YOUR_DOCKERHUB_USERNAME"
IMAGE_NAME="clinicalner"
TAG="latest"
EC2_KEY="your-key-pair-name"       # AWS EC2 key pair name (no .pem)
EC2_KEY_FILE="~/.ssh/${EC2_KEY}.pem"
REGION="us-east-1"
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════"
echo "  ClinicalNER — AWS EC2 Deployment"
echo "════════════════════════════════════════════════════"

# Step 1: Build and push Docker image to DockerHub
echo ""
echo "► Step 1: Building Docker image..."
docker build -t ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG} \
    -f docker/Dockerfile .

echo "► Step 2: Pushing to DockerHub..."
docker push ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}
echo "  ✓ Image pushed: ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"

# Step 2: Launch EC2 t2.micro (free tier)
echo ""
echo "► Step 3: Launching EC2 t2.micro instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type t2.micro \
    --key-name ${EC2_KEY} \
    --security-group-ids $(aws ec2 create-security-group \
        --group-name clinicalner-sg \
        --description "ClinicalNER Flask API" \
        --query 'GroupId' --output text 2>/dev/null || \
        aws ec2 describe-security-groups \
        --group-names clinicalner-sg \
        --query 'SecurityGroups[0].GroupId' --output text) \
    --user-data "$(cat << 'USERDATA'
#!/bin/bash
yum update -y
amazon-linux-extras install docker -y
service docker start
usermod -a -G docker ec2-user
docker pull ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}
docker run -d -p 5000:5000 --restart always \
    --name clinicalner \
    ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}
USERDATA
)" \
    --query 'Instances[0].InstanceId' \
    --output text \
    --region ${REGION})

echo "  ✓ Instance launched: ${INSTANCE_ID}"

# Open port 5000 (and optionally 80 via nginx proxy)
aws ec2 authorize-security-group-ingress \
    --group-name clinicalner-sg \
    --protocol tcp --port 5000 --cidr 0.0.0.0/0 \
    --region ${REGION} 2>/dev/null || true

# Step 3: Wait for instance and get public IP
echo ""
echo "► Step 4: Waiting for instance to start (60s)..."
sleep 60
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids ${INSTANCE_ID} \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text \
    --region ${REGION})

echo ""
echo "════════════════════════════════════════════════════"
echo "  Deployment complete!"
echo "────────────────────────────────────────────────────"
echo "  Instance ID : ${INSTANCE_ID}"
echo "  Public IP   : ${PUBLIC_IP}"
echo ""
echo "  Dashboard   : http://${PUBLIC_IP}:5000/dashboard"
echo "  API         : http://${PUBLIC_IP}:5000/api/deidentify"
echo "  Health      : http://${PUBLIC_IP}:5000/health"
echo ""
echo "  SSH access  : ssh -i ${EC2_KEY_FILE} ec2-user@${PUBLIC_IP}"
echo "════════════════════════════════════════════════════"
echo ""

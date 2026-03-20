# AWS Deployment Guide

## Overview
This guide covers deploying the ClinicalNER pipeline to AWS using ECS Fargate with Terraform.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AWS Cloud                                │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Application Load Balancer (ALB)                       │ │
│  │  - Health checks                                       │ │
│  │  - SSL termination                                     │ │
│  └──────────────────┬─────────────────────────────────────┘ │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────────────────┐│
│  │  ECS Fargate Cluster                                    ││
│  │  ┌──────────────┐  ┌──────────────┐                    ││
│  │  │ Task 1       │  │ Task 2       │                    ││
│  │  │ ClinicalNER  │  │ ClinicalNER  │                    ││
│  │  │ Container    │  │ Container    │                    ││
│  │  └──────┬───────┘  └──────┬───────┘                    ││
│  │         │                  │                            ││
│  │         └────────┬─────────┘                            ││
│  └──────────────────┼──────────────────────────────────────┘│
│                     │                                        │
│  ┌──────────────────▼──────────────────────────────────────┐│
│  │  Amazon EFS (Elastic File System)                       ││
│  │  - Persistent data storage                              ││
│  │  - Shared across tasks                                  ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Amazon ECR (Elastic Container Registry)                ││
│  │  - Docker image storage                                 ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  CloudWatch Logs                                        ││
│  │  - Application logs                                     ││
│  │  - Metrics and monitoring                               ││
│  └─────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured
3. **Terraform** >= 1.0 installed
4. **Docker** installed locally

## Step 1: Configure AWS Credentials

```bash
# Configure AWS CLI
aws configure

# Verify credentials
aws sts get-caller-identity
```

## Step 2: Build and Push Docker Image

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Build image
docker build -t clinicalner:latest -f docker/Dockerfile .

# Tag image
docker tag clinicalner:latest \
  <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/clinicalner:latest

# Push to ECR
docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/clinicalner:latest
```

## Step 3: Deploy Infrastructure with Terraform

```bash
# Initialize Terraform
cd terraform
terraform init

# Review planned changes
terraform plan

# Apply infrastructure
terraform apply

# Note the outputs:
# - alb_dns_name: Load balancer URL
# - ecr_repository_url: Docker registry URL
# - ecs_cluster_name: ECS cluster name
```

## Step 4: Verify Deployment

```bash
# Get ALB DNS name
ALB_DNS=$(terraform output -raw alb_dns_name)

# Test health endpoint
curl http://$ALB_DNS/health

# Test dashboard
open http://$ALB_DNS/dashboard
```

## Step 5: Deploy Using Docker Compose (Alternative)

```bash
# Set environment variables
export AWS_ACCOUNT_ID=<your-account-id>
export AWS_REGION=us-east-1
export EFS_DNS_NAME=<efs-dns-name>

# Deploy
docker-compose -f docker-compose.aws.yml up -d

# View logs
docker-compose -f docker-compose.aws.yml logs -f
```

## Monitoring

### CloudWatch Logs

```bash
# View logs
aws logs tail /ecs/clinicalner --follow

# Filter for errors
aws logs filter-log-events \
  --log-group-name /ecs/clinicalner \
  --filter-pattern "ERROR"
```

### CloudWatch Metrics

Key metrics to monitor:
- **CPUUtilization**: Target < 70%
- **MemoryUtilization**: Target < 80%
- **TargetResponseTime**: Target < 500ms
- **HealthyHostCount**: Should equal desired count

### Alarms

```bash
# Create CPU alarm
aws cloudwatch put-metric-alarm \
  --alarm-name clinicalner-high-cpu \
  --alarm-description "Alert when CPU exceeds 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2
```

## Scaling

### Auto Scaling

```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/clinicalner-cluster/clinicalner-service \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 2 \
  --max-capacity 10

# Create scaling policy
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --resource-id service/clinicalner-cluster/clinicalner-service \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-name cpu-scaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration \
    '{"TargetValue":70.0,"PredefinedMetricSpecification":{"PredefinedMetricType":"ECSServiceAverageCPUUtilization"}}'
```

## Security

### SSL/TLS Configuration

```bash
# Request ACM certificate
aws acm request-certificate \
  --domain-name clinicalner.example.com \
  --validation-method DNS

# Update ALB listener to use HTTPS
# (Add to terraform/main.tf)
```

### Secrets Management

```bash
# Store database credentials in Secrets Manager
aws secretsmanager create-secret \
  --name clinicalner/db-credentials \
  --secret-string '{"username":"admin","password":"<secure-password>"}'

# Update ECS task definition to use secrets
# (Add to terraform/main.tf)
```

## Cost Optimization

### Estimated Monthly Costs (us-east-1)

| Service | Configuration | Monthly Cost |
|---------|--------------|--------------|
| ECS Fargate | 2 tasks, 2 vCPU, 4GB RAM | ~$120 |
| ALB | 1 load balancer | ~$20 |
| EFS | 10GB storage | ~$3 |
| ECR | 5GB storage | ~$0.50 |
| CloudWatch | Logs + metrics | ~$10 |
| **Total** | | **~$153/month** |

### Cost Reduction Tips

1. **Use Spot Instances** for non-production
2. **Enable EFS Lifecycle Management** to move old data to IA storage
3. **Set CloudWatch log retention** to 30 days
4. **Use Reserved Capacity** for predictable workloads

## Backup and Disaster Recovery

### EFS Backup

```bash
# Enable automatic backups
aws backup create-backup-plan \
  --backup-plan file://backup-plan.json

# Manual backup
aws efs create-backup \
  --file-system-id <efs-id>
```

### Database Backup

```bash
# Export database to S3
aws s3 cp data/clinicalner.db s3://clinicalner-backups/$(date +%Y%m%d)/
```

## Troubleshooting

### Task Fails to Start

```bash
# Check task logs
aws ecs describe-tasks \
  --cluster clinicalner-cluster \
  --tasks <task-id>

# View stopped task reason
aws ecs describe-tasks \
  --cluster clinicalner-cluster \
  --tasks <task-id> \
  --query 'tasks[0].stoppedReason'
```

### Health Check Failures

```bash
# Test health endpoint directly
aws ecs execute-command \
  --cluster clinicalner-cluster \
  --task <task-id> \
  --container clinicalner \
  --interactive \
  --command "curl localhost:5000/health"
```

### High Memory Usage

```bash
# Check memory metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name MemoryUtilization \
  --dimensions Name=ServiceName,Value=clinicalner-service \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average
```

## Cleanup

```bash
# Destroy all infrastructure
cd terraform
terraform destroy

# Delete ECR images
aws ecr batch-delete-image \
  --repository-name clinicalner \
  --image-ids imageTag=latest

# Delete CloudWatch logs
aws logs delete-log-group --log-group-name /ecs/clinicalner
```

## Production Checklist

- [ ] SSL/TLS certificate configured
- [ ] Secrets stored in AWS Secrets Manager
- [ ] Auto-scaling policies configured
- [ ] CloudWatch alarms set up
- [ ] Backup strategy implemented
- [ ] Monitoring dashboard created
- [ ] Security groups reviewed
- [ ] IAM roles follow least privilege
- [ ] Cost alerts configured
- [ ] Disaster recovery plan documented

## Support

For deployment issues:
- **AWS Support**: https://console.aws.amazon.com/support
- **Terraform Docs**: https://registry.terraform.io/providers/hashicorp/aws
- **Project Issues**: https://github.com/ansh-0069/ClinicalNER/issues

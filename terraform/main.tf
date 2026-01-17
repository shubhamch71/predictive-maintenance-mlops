# # =============================================================================
# # Terraform Configuration for Predictive Maintenance MLOps Platform
# # =============================================================================
# # This Terraform configuration creates:
# # - VPC with public/private subnets
# # - EKS cluster with managed node groups
# # - IAM roles and policies
# # - IRSA for workloads (Prometheus, Grafana, ml-pipeline, etc.)
# # - Jenkins EC2 instance with EKS access
# # - EBS CSI driver for persistent storage
# # - S3 bucket for ML artifacts
# # - Secrets Manager for sensitive data
# # =============================================================================

# terraform {
#   required_version = ">= 1.5.0"

#   required_providers {
#     aws = {
#       source  = "hashicorp/aws"
#       version = "~> 5.0"
#     }
#     kubernetes = {
#       source  = "hashicorp/kubernetes"
#       version = "~> 2.23"
#     }
#     helm = {
#       source  = "hashicorp/helm"
#       version = "~> 2.11"
#     }
#     tls = {
#       source  = "hashicorp/tls"
#       version = "~> 4.0"
#     }
#   }

#   # Uncomment to use S3 backend for state management
#   # backend "s3" {
#   #   bucket         = "your-terraform-state-bucket"
#   #   key            = "predictive-maintenance/terraform.tfstate"
#   #   region         = "us-west-2"
#   #   dynamodb_table = "terraform-locks"
#   #   encrypt        = true
#   # }
# }

# # =============================================================================
# # Provider Configuration
# # =============================================================================

# provider "aws" {
#   region = var.aws_region

#   default_tags {
#     tags = {
#       Project     = var.project_name
#       Environment = var.environment
#       ManagedBy   = "terraform"
#     }
#   }
# }

# # Kubernetes provider configured after EKS cluster is created
# provider "kubernetes" {
#   host                   = module.eks.cluster_endpoint
#   cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

#   exec {
#     api_version = "client.authentication.k8s.io/v1beta1"
#     command     = "aws"
#     args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
#   }
# }

# provider "helm" {
#   kubernetes {
#     host                   = module.eks.cluster_endpoint
#     cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

#     exec {
#       api_version = "client.authentication.k8s.io/v1beta1"
#       command     = "aws"
#       args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
#     }
#   }
# }

# # =============================================================================
# # Data Sources
# # =============================================================================

# data "aws_availability_zones" "available" {
#   state = "available"
# }

# data "aws_caller_identity" "current" {}

# # =============================================================================
# # VPC Module
# # =============================================================================

# module "vpc" {
#   source  = "terraform-aws-modules/vpc/aws"
#   version = "~> 5.0"

#   name = "${var.project_name}-vpc"
#   cidr = var.vpc_cidr

#   azs             = slice(data.aws_availability_zones.available.names, 0, 3)
#   private_subnets = var.private_subnet_cidrs
#   public_subnets  = var.public_subnet_cidrs

#   # NAT Gateway configuration
#   enable_nat_gateway     = true
#   single_nat_gateway     = var.environment == "dev" ? true : false
#   one_nat_gateway_per_az = var.environment == "production" ? true : false

#   # DNS settings
#   enable_dns_hostnames = true
#   enable_dns_support   = true

#   # Kubernetes tags for subnet discovery
#   public_subnet_tags = {
#     "kubernetes.io/role/elb"                    = 1
#     "kubernetes.io/cluster/${var.cluster_name}" = "shared"
#   }

#   private_subnet_tags = {
#     "kubernetes.io/role/internal-elb"           = 1
#     "kubernetes.io/cluster/${var.cluster_name}" = "shared"
#   }

#   tags = {
#     Name = "${var.project_name}-vpc"
#   }
# }

# # =============================================================================
# # EKS Cluster Module
# # =============================================================================

# module "eks" {
#   source  = "terraform-aws-modules/eks/aws"
#   version = "~> 19.0"

#   cluster_name    = var.cluster_name
#   cluster_version = var.cluster_version

#   # VPC Configuration
#   vpc_id     = module.vpc.vpc_id
#   subnet_ids = module.vpc.private_subnets

#   # Cluster endpoint configuration
#   cluster_endpoint_public_access  = true
#   cluster_endpoint_private_access = true

#   # Enable IRSA
#   enable_irsa = true

#   # Cluster addons
#   cluster_addons = {
#     coredns = {
#       most_recent = true
#     }
#     kube-proxy = {
#       most_recent = true
#     }
#     vpc-cni = {
#       most_recent              = true
#       before_compute           = true
#       service_account_role_arn = module.vpc_cni_irsa.iam_role_arn
#       configuration_values = jsonencode({
#         env = {
#           ENABLE_PREFIX_DELEGATION = "true"
#           WARM_PREFIX_TARGET       = "1"
#         }
#       })
#     }
#     aws-ebs-csi-driver = {
#       most_recent              = true
#       service_account_role_arn = module.ebs_csi_irsa.iam_role_arn
#     }
#   }

#   # Managed node groups
#   eks_managed_node_groups = {
#     # General purpose nodes
#     general = {
#       name           = "${var.cluster_name}-general"
#       instance_types = var.node_instance_types
#       capacity_type  = "ON_DEMAND"

#       min_size     = var.node_min_size
#       max_size     = var.node_max_size
#       desired_size = var.node_desired_size

#       labels = {
#         role = "general"
#       }

#       tags = {
#         NodeGroup = "general"
#       }
#     }

#     # ML workloads nodes (optional - larger instances)
#     ml_workloads = {
#       name           = "${var.cluster_name}-ml"
#       instance_types = var.ml_node_instance_types
#       capacity_type  = "ON_DEMAND"

#       min_size     = 0
#       max_size     = var.ml_node_max_size
#       desired_size = 0

#       labels = {
#         role     = "ml-workloads"
#         workload = "ml-training"
#       }

#       taints = [
#         {
#           key    = "workload"
#           value  = "ml-training"
#           effect = "NO_SCHEDULE"
#         }
#       ]

#       tags = {
#         NodeGroup = "ml-workloads"
#       }
#     }
#   }

#   # aws-auth configmap
#   manage_aws_auth_configmap = true

#   aws_auth_roles = [
#     {
#       rolearn  = module.jenkins_ec2.jenkins_role_arn
#       username = "jenkins"
#       groups   = ["system:masters"]
#     },
#   ]

#   aws_auth_users = var.eks_admin_users

#   tags = {
#     Name = var.cluster_name
#   }
# }

# # =============================================================================
# # IRSA Modules (IAM Roles for Service Accounts)
# # =============================================================================

# # VPC CNI IRSA
# module "vpc_cni_irsa" {
#   source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
#   version = "~> 5.0"

#   role_name             = "${var.cluster_name}-vpc-cni"
#   attach_vpc_cni_policy = true
#   vpc_cni_enable_ipv4   = true

#   oidc_providers = {
#     main = {
#       provider_arn               = module.eks.oidc_provider_arn
#       namespace_service_accounts = ["kube-system:aws-node"]
#     }
#   }
# }

# # EBS CSI Driver IRSA
# module "ebs_csi_irsa" {
#   source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
#   version = "~> 5.0"

#   role_name             = "${var.cluster_name}-ebs-csi"
#   attach_ebs_csi_policy = true

#   oidc_providers = {
#     main = {
#       provider_arn               = module.eks.oidc_provider_arn
#       namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
#     }
#   }
# }

# # Prometheus IRSA
# module "prometheus_irsa" {
#   source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
#   version = "~> 5.0"

#   role_name = "${var.cluster_name}-prometheus"

#   role_policy_arns = {
#     cloudwatch = "arn:aws:iam::aws:policy/CloudWatchReadOnlyAccess"
#   }

#   oidc_providers = {
#     main = {
#       provider_arn               = module.eks.oidc_provider_arn
#       namespace_service_accounts = ["mlops:prometheus", "monitoring:prometheus"]
#     }
#   }
# }

# # Grafana IRSA
# module "grafana_irsa" {
#   source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
#   version = "~> 5.0"

#   role_name = "${var.cluster_name}-grafana"

#   role_policy_arns = {
#     cloudwatch = "arn:aws:iam::aws:policy/CloudWatchReadOnlyAccess"
#   }

#   oidc_providers = {
#     main = {
#       provider_arn               = module.eks.oidc_provider_arn
#       namespace_service_accounts = ["mlops:grafana", "monitoring:grafana"]
#     }
#   }
# }

# # MLflow IRSA (S3 access for artifacts)
# module "mlflow_irsa" {
#   source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
#   version = "~> 5.0"

#   role_name = "${var.cluster_name}-mlflow"

#   role_policy_arns = {
#     s3 = aws_iam_policy.mlflow_s3.arn
#   }

#   oidc_providers = {
#     main = {
#       provider_arn               = module.eks.oidc_provider_arn
#       namespace_service_accounts = ["mlops:mlflow"]
#     }
#   }
# }

# # Kubeflow ml-pipeline IRSA
# module "ml_pipeline_irsa" {
#   source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
#   version = "~> 5.0"

#   role_name = "${var.cluster_name}-ml-pipeline"

#   role_policy_arns = {
#     s3      = aws_iam_policy.ml_pipeline_s3.arn
#     secrets = aws_iam_policy.ml_pipeline_secrets.arn
#   }

#   oidc_providers = {
#     main = {
#       provider_arn               = module.eks.oidc_provider_arn
#       namespace_service_accounts = ["kubeflow:ml-pipeline", "kubeflow:pipeline-runner"]
#     }
#   }
# }

# # KServe IRSA (for model loading from S3)
# module "kserve_irsa" {
#   source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
#   version = "~> 5.0"

#   role_name = "${var.cluster_name}-kserve"

#   role_policy_arns = {
#     s3 = aws_iam_policy.kserve_s3.arn
#   }

#   oidc_providers = {
#     main = {
#       provider_arn               = module.eks.oidc_provider_arn
#       namespace_service_accounts = ["mlops:default", "kserve-serving:default"]
#     }
#   }
# }

# # =============================================================================
# # IAM Policies for IRSA
# # =============================================================================

# # S3 bucket for ML artifacts
# resource "aws_s3_bucket" "ml_artifacts" {
#   bucket = "${var.project_name}-ml-artifacts-${data.aws_caller_identity.current.account_id}"

#   tags = {
#     Name = "${var.project_name}-ml-artifacts"
#   }
# }

# resource "aws_s3_bucket_versioning" "ml_artifacts" {
#   bucket = aws_s3_bucket.ml_artifacts.id
#   versioning_configuration {
#     status = "Enabled"
#   }
# }

# resource "aws_s3_bucket_server_side_encryption_configuration" "ml_artifacts" {
#   bucket = aws_s3_bucket.ml_artifacts.id

#   rule {
#     apply_server_side_encryption_by_default {
#       sse_algorithm = "aws:kms"
#     }
#   }
# }

# # MLflow S3 policy
# resource "aws_iam_policy" "mlflow_s3" {
#   name        = "${var.cluster_name}-mlflow-s3"
#   description = "IAM policy for MLflow to access S3"

#   policy = jsonencode({
#     Version = "2012-10-17"
#     Statement = [
#       {
#         Effect = "Allow"
#         Action = [
#           "s3:GetObject",
#           "s3:PutObject",
#           "s3:DeleteObject",
#           "s3:ListBucket"
#         ]
#         Resource = [
#           aws_s3_bucket.ml_artifacts.arn,
#           "${aws_s3_bucket.ml_artifacts.arn}/*"
#         ]
#       }
#     ]
#   })
# }

# # ML Pipeline S3 policy
# resource "aws_iam_policy" "ml_pipeline_s3" {
#   name        = "${var.cluster_name}-ml-pipeline-s3"
#   description = "IAM policy for ml-pipeline to access S3"

#   policy = jsonencode({
#     Version = "2012-10-17"
#     Statement = [
#       {
#         Effect = "Allow"
#         Action = [
#           "s3:GetObject",
#           "s3:PutObject",
#           "s3:DeleteObject",
#           "s3:ListBucket"
#         ]
#         Resource = [
#           aws_s3_bucket.ml_artifacts.arn,
#           "${aws_s3_bucket.ml_artifacts.arn}/*"
#         ]
#       }
#     ]
#   })
# }

# # ML Pipeline Secrets Manager policy
# resource "aws_iam_policy" "ml_pipeline_secrets" {
#   name        = "${var.cluster_name}-ml-pipeline-secrets"
#   description = "IAM policy for ml-pipeline to access Secrets Manager"

#   policy = jsonencode({
#     Version = "2012-10-17"
#     Statement = [
#       {
#         Effect = "Allow"
#         Action = [
#           "secretsmanager:GetSecretValue",
#           "secretsmanager:DescribeSecret"
#         ]
#         Resource = [
#           "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:${var.project_name}/*"
#         ]
#       }
#     ]
#   })
# }

# # KServe S3 policy (read-only for model loading)
# resource "aws_iam_policy" "kserve_s3" {
#   name        = "${var.cluster_name}-kserve-s3"
#   description = "IAM policy for KServe to load models from S3"

#   policy = jsonencode({
#     Version = "2012-10-17"
#     Statement = [
#       {
#         Effect = "Allow"
#         Action = [
#           "s3:GetObject",
#           "s3:ListBucket"
#         ]
#         Resource = [
#           aws_s3_bucket.ml_artifacts.arn,
#           "${aws_s3_bucket.ml_artifacts.arn}/*"
#         ]
#       }
#     ]
#   })
# }

# # =============================================================================
# # Jenkins EC2 Instance
# # =============================================================================

# module "jenkins_ec2" {
#   source = "./modules/jenkins-ec2"

#   project_name = var.project_name
#   environment  = var.environment

#   vpc_id            = module.vpc.vpc_id
#   subnet_id         = module.vpc.public_subnets[0]
#   instance_type     = var.jenkins_instance_type
#   key_name          = var.jenkins_key_name
#   allowed_ssh_cidrs = var.jenkins_allowed_ssh_cidrs

#   eks_cluster_name     = module.eks.cluster_name
#   eks_cluster_endpoint = module.eks.cluster_endpoint
#   eks_cluster_arn      = module.eks.cluster_arn

#   tags = {
#     Name = "${var.project_name}-jenkins"
#   }
# }

# # =============================================================================
# # Storage Class for EKS
# # =============================================================================

# resource "kubernetes_storage_class" "gp3" {
#   metadata {
#     name = "gp3"
#     annotations = {
#       "storageclass.kubernetes.io/is-default-class" = "true"
#     }
#   }

#   storage_provisioner    = "ebs.csi.aws.com"
#   volume_binding_mode    = "WaitForFirstConsumer"
#   allow_volume_expansion = true

#   parameters = {
#     type      = "gp3"
#     fsType    = "ext4"
#     encrypted = "true"
#   }

#   depends_on = [module.eks]
# }

# # =============================================================================
# # AWS Secrets Manager for sensitive data
# # =============================================================================

# resource "aws_secretsmanager_secret" "postgres_credentials" {
#   name        = "${var.project_name}/postgres-credentials"
#   description = "PostgreSQL credentials for MLflow"

#   tags = {
#     Name = "${var.project_name}-postgres-credentials"
#   }
# }

# resource "aws_secretsmanager_secret_version" "postgres_credentials" {
#   secret_id = aws_secretsmanager_secret.postgres_credentials.id
#   secret_string = jsonencode({
#     username = "mlflow"
#     password = var.postgres_password
#     database = "mlflowdb"
#   })
# }

# resource "aws_secretsmanager_secret" "kubeflow_mysql_credentials" {
#   name        = "${var.project_name}/kubeflow-mysql-credentials"
#   description = "MySQL credentials for Kubeflow Pipelines"

#   tags = {
#     Name = "${var.project_name}-kubeflow-mysql-credentials"
#   }
# }

# resource "aws_secretsmanager_secret_version" "kubeflow_mysql_credentials" {
#   secret_id = aws_secretsmanager_secret.kubeflow_mysql_credentials.id
#   secret_string = jsonencode({
#     username = "root"
#     password = var.mysql_password
#     database = "mlpipeline"
#   })
# }

# # =============================================================================
# # Outputs
# # =============================================================================

# output "vpc_id" {
#   description = "VPC ID"
#   value       = module.vpc.vpc_id
# }

# output "eks_cluster_name" {
#   description = "EKS cluster name"
#   value       = module.eks.cluster_name
# }

# output "eks_cluster_endpoint" {
#   description = "EKS cluster endpoint"
#   value       = module.eks.cluster_endpoint
# }

# output "eks_cluster_arn" {
#   description = "EKS cluster ARN"
#   value       = module.eks.cluster_arn
# }

# output "eks_oidc_provider_arn" {
#   description = "EKS OIDC provider ARN"
#   value       = module.eks.oidc_provider_arn
# }

# output "jenkins_public_ip" {
#   description = "Jenkins EC2 public IP"
#   value       = module.jenkins_ec2.public_ip
# }

# output "jenkins_url" {
#   description = "Jenkins URL"
#   value       = "http://${module.jenkins_ec2.public_ip}:8080"
# }

# output "s3_artifacts_bucket" {
#   description = "S3 bucket for ML artifacts"
#   value       = aws_s3_bucket.ml_artifacts.bucket
# }

# output "kubeconfig_command" {
#   description = "Command to configure kubectl"
#   value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
# }

# # IRSA Role ARNs (for ServiceAccount annotations)
# output "irsa_roles" {
#   description = "IRSA role ARNs for service account annotations"
#   value = {
#     prometheus  = module.prometheus_irsa.iam_role_arn
#     grafana     = module.grafana_irsa.iam_role_arn
#     mlflow      = module.mlflow_irsa.iam_role_arn
#     ml_pipeline = module.ml_pipeline_irsa.iam_role_arn
#     kserve      = module.kserve_irsa.iam_role_arn
#   }
# }


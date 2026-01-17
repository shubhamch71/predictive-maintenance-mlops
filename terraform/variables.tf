# =============================================================================
# Terraform Variables for Predictive Maintenance MLOps Platform
# =============================================================================

# -----------------------------------------------------------------------------
# General Configuration
# -----------------------------------------------------------------------------

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "predictive-maintenance"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "production"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

# -----------------------------------------------------------------------------
# VPC Configuration
# -----------------------------------------------------------------------------

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.20.0/24", "10.0.30.0/24"]
}

# -----------------------------------------------------------------------------
# EKS Configuration
# -----------------------------------------------------------------------------

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "pm-mlops-cluster"
}

variable "cluster_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

# General node group configuration
variable "node_instance_types" {
  description = "Instance types for general node group"
  type        = list(string)
  default     = ["t3.large", "t3.xlarge"]
}

variable "node_min_size" {
  description = "Minimum size of general node group"
  type        = number
  default     = 2
}

variable "node_max_size" {
  description = "Maximum size of general node group"
  type        = number
  default     = 5
}

variable "node_desired_size" {
  description = "Desired size of general node group"
  type        = number
  default     = 3
}

# ML workloads node group configuration
variable "ml_node_instance_types" {
  description = "Instance types for ML workloads node group"
  type        = list(string)
  default     = ["m5.2xlarge", "m5.4xlarge"]
}

variable "ml_node_max_size" {
  description = "Maximum size of ML workloads node group"
  type        = number
  default     = 3
}

# EKS admin users
variable "eks_admin_users" {
  description = "List of IAM users to add to aws-auth ConfigMap"
  type = list(object({
    userarn  = string
    username = string
    groups   = list(string)
  }))
  default = []
}

# -----------------------------------------------------------------------------
# Jenkins EC2 Configuration
# -----------------------------------------------------------------------------

variable "jenkins_instance_type" {
  description = "Instance type for Jenkins EC2"
  type        = string
  default     = "t3.medium"
}

variable "jenkins_key_name" {
  description = "SSH key pair name for Jenkins EC2"
  type        = string
  default     = ""
}

variable "jenkins_allowed_ssh_cidrs" {
  description = "CIDR blocks allowed to SSH to Jenkins"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# -----------------------------------------------------------------------------
# Database Credentials (use AWS Secrets Manager in production)
# -----------------------------------------------------------------------------

variable "postgres_password" {
  description = "Password for PostgreSQL (change in production!)"
  type        = string
  default     = "mlflow_password_change_me"
  sensitive   = true
}

variable "mysql_password" {
  description = "Password for Kubeflow MySQL (change in production!)"
  type        = string
  default     = "pipelines_password_change_me"
  sensitive   = true
}

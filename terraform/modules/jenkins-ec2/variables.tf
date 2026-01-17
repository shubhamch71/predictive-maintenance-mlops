# =============================================================================
# Jenkins EC2 Module Variables
# =============================================================================

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID for Jenkins EC2"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.medium"
}

variable "key_name" {
  description = "SSH key pair name"
  type        = string
  default     = ""
}

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed to SSH"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "eks_cluster_name" {
  description = "EKS cluster name"
  type        = string
}

variable "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  type        = string
}

variable "eks_cluster_arn" {
  description = "EKS cluster ARN"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}


# =============================================================================
# Jenkins EC2 Instance Module
# =============================================================================
# Creates an EC2 instance with:
# - Jenkins pre-installed via user_data
# - kubectl, aws-cli, eksctl, helm installed
# - IAM role with EKS access
# - Security group for SSH and Jenkins UI
# =============================================================================

# -----------------------------------------------------------------------------
# Data Sources
# -----------------------------------------------------------------------------

data "aws_ami" "amazon_linux_2" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

data "aws_region" "current" {}

# -----------------------------------------------------------------------------
# IAM Role for Jenkins EC2
# -----------------------------------------------------------------------------

resource "aws_iam_role" "jenkins" {
  name = "${var.project_name}-jenkins-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# IAM policy for EKS access
resource "aws_iam_policy" "jenkins_eks" {
  name        = "${var.project_name}-jenkins-eks-policy"
  description = "IAM policy for Jenkins to access EKS"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "eks:DescribeCluster",
          "eks:ListClusters",
          "eks:AccessKubernetesApi",
          "eks:DescribeNodegroup",
          "eks:ListNodegroups"
        ]
        Resource = [
          var.eks_cluster_arn,
          "${var.eks_cluster_arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:DescribeRepositories",
          "ecr:CreateRepository"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = "*"
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "jenkins_eks" {
  role       = aws_iam_role.jenkins.name
  policy_arn = aws_iam_policy.jenkins_eks.arn
}

# SSM access for Session Manager (alternative to SSH)
resource "aws_iam_role_policy_attachment" "jenkins_ssm" {
  role       = aws_iam_role.jenkins.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "jenkins" {
  name = "${var.project_name}-jenkins-profile"
  role = aws_iam_role.jenkins.name

  tags = var.tags
}

# -----------------------------------------------------------------------------
# Security Group for Jenkins
# -----------------------------------------------------------------------------

resource "aws_security_group" "jenkins" {
  name        = "${var.project_name}-jenkins-sg"
  description = "Security group for Jenkins EC2"
  vpc_id      = var.vpc_id

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
    description = "SSH access"
  }

  # Jenkins UI
  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Jenkins UI"
  }

  # HTTPS (for webhooks)
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS"
  }

  # Allow all outbound
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }

  tags = merge(var.tags, {
    Name = "${var.project_name}-jenkins-sg"
  })
}

# -----------------------------------------------------------------------------
# Jenkins EC2 Instance
# -----------------------------------------------------------------------------

resource "aws_instance" "jenkins" {
  ami                    = data.aws_ami.amazon_linux_2.id
  instance_type          = var.instance_type
  key_name               = var.key_name != "" ? var.key_name : null
  subnet_id              = var.subnet_id
  vpc_security_group_ids = [aws_security_group.jenkins.id]
  iam_instance_profile   = aws_iam_instance_profile.jenkins.name

  associate_public_ip_address = true

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 50
    encrypted             = true
    delete_on_termination = true
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    aws_region       = data.aws_region.current.name
    eks_cluster_name = var.eks_cluster_name
  }))

  tags = merge(var.tags, {
    Name = "${var.project_name}-jenkins"
  })

  lifecycle {
    ignore_changes = [ami]
  }
}

# Elastic IP for consistent access
resource "aws_eip" "jenkins" {
  instance = aws_instance.jenkins.id
  domain   = "vpc"

  tags = merge(var.tags, {
    Name = "${var.project_name}-jenkins-eip"
  })
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "instance_id" {
  description = "Jenkins EC2 instance ID"
  value       = aws_instance.jenkins.id
}

output "public_ip" {
  description = "Jenkins EC2 public IP"
  value       = aws_eip.jenkins.public_ip
}

output "jenkins_role_arn" {
  description = "Jenkins IAM role ARN"
  value       = aws_iam_role.jenkins.arn
}

output "security_group_id" {
  description = "Jenkins security group ID"
  value       = aws_security_group.jenkins.id
}

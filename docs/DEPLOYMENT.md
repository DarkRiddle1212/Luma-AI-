# Deployment Guide for Luma AI System

## Overview

This guide provides step-by-step instructions for deploying the Luma AI system in production environments. Luma is designed as a local-first personal AI system but can be deployed on servers for remote access.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Deployment Options](#deployment-options)
3. [Local Development Deployment](#local-development-deployment)
4. [Production Server Deployment](#production-server-deployment)
5. [Docker Deployment](#docker-deployment-optional)
6. [Post-Deployment Verification](#post-deployment-verification)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+, Debian 11+), macOS 10.15+, or Windows 10+
- **Python**: 3.9 or higher
- **Memory**: Minimum 512MB RAM, recommended 1GB+
- **Disk Space**: Minimum 100MB for application, additional space for database
- **Network**: Internet connection for initial setup and dependency installation

### Required Software

- Python 3.9+
- pip (Python package manager)
- virtualenv or venv
- Git (for cloning repository)

### Optional Software

- Nginx or Caddy (reverse proxy for production)
- systemd (for service management on Linux)
- Docker (for containerized deployment)

## Deployment Options

### Option 1: Local Development
- Quick setup for testing and development
- Runs on localhost only
- No HTTPS, no authentication
- **Use case**: Local development, testing

### Option 2: Production Server
- Full production setup with reverse proxy
- HTTPS with SSL/TLS certificates
- Systemd service for automatic startup
- **Use case**: Personal server, remote access

### Option 3: Docker (Optional)
- Containerized deployment
- Portable and reproducible
- Easy scaling and management
- **Use case**: Cloud deployment, multi-instance

## Local Development Deployment

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd luma
```

### Step 2: Create Virtual Environment

```bash
# Using venv (Python 3.3+)
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# For local development, defaults are usually fine
nano .env  # or use your preferred editor
```

### Step 5: Initialize Database

```bash
# Database tables are created automatically on first run
# Optionally, verify database directory exists
mkdir -p data
```

### Step 6: Run Application

```bash
# Development mode with auto-reload
uvicorn luma.main:app --reload --host 127.0.0.1 --port 8000

# Or using Python module
python -m uvicorn luma.main:app --reload
```

### Step 7: Verify Installation

Open browser and navigate to:
- Application: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Production Server Deployment

### Step 1: Prepare Server

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3 python3-pip python3-venv nginx certbot python3-certbot-nginx

# Create application user (security best practice)
sudo useradd -m -s /bin/bash luma
sudo su - luma
```

### Step 2: Deploy Application

```bash
# Clone repository
cd /home/luma
git clone <repository-url> luma-app
cd luma-app

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configure Production Environment

```bash
# Create production environment file
cp .env.example .env

# Edit with production settings
nano .env
```

**Production .env example:**
```env
# Database
DATABASE_URL=sqlite:////home/luma/luma-app/data/luma.db

# API Configuration
API_HOST=127.0.0.1
API_PORT=8000
API_PREFIX=/api/v1

# Logging
LOG_LEVEL=INFO
ENVIRONMENT=production

# Add authentication (implement API key middleware first)
# API_KEY=your-secure-api-key-here
```

### Step 4: Set File Permissions

```bash
# Ensure proper permissions
chmod 600 .env
mkdir -p data logs
chmod 700 data logs

# Database file permissions (after first run)
chmod 600 data/luma.db
```

### Step 5: Create Systemd Service

Exit from luma user and create service file as root:

```bash
exit  # Exit from luma user
sudo nano /etc/systemd/system/luma.service
```

**Service file content:**
```ini
[Unit]
Description=Luma AI System
After=network.target

[Service]
Type=simple
User=luma
Group=luma
WorkingDirectory=/home/luma/luma-app
Environment="PATH=/home/luma/luma-app/venv/bin"
ExecStart=/home/luma/luma-app/venv/bin/uvicorn luma.main:app --host 127.0.0.1 --port 8000 --workers 4
Restart=always
RestartSec=10

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/home/luma/luma-app/data /home/luma/luma-app/logs

[Install]
WantedBy=multi-user.target
```

### Step 6: Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable luma

# Start service
sudo systemctl start luma

# Check status
sudo systemctl status luma

# View logs
sudo journalctl -u luma -f
```

### Step 7: Configure Nginx Reverse Proxy

```bash
sudo nano /etc/nginx/sites-available/luma
```

**Nginx configuration:**
```nginx
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain

    # Redirect HTTP to HTTPS (after SSL setup)
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;  # Replace with your domain

    # SSL certificates (will be configured by certbot)
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Logging
    access_log /var/log/nginx/luma_access.log;
    error_log /var/log/nginx/luma_error.log;

    # Proxy settings
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Rate limiting (optional)
    limit_req_zone $binary_remote_addr zone=luma_limit:10m rate=10r/s;
    limit_req zone=luma_limit burst=20 nodelay;
}
```

### Step 8: Enable Nginx Site

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/luma /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

### Step 9: Setup SSL/TLS with Let's Encrypt

```bash
# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Certbot will automatically configure Nginx
# Follow the prompts to complete setup

# Test automatic renewal
sudo certbot renew --dry-run
```

### Step 10: Configure Firewall

```bash
# Allow SSH (if not already allowed)
sudo ufw allow ssh

# Allow HTTP and HTTPS
sudo ufw allow 'Nginx Full'

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status
```

## Docker Deployment (Optional)

### Step 1: Create Dockerfile

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY luma/ ./luma/
COPY .env.example .env

# Create data directory
RUN mkdir -p /app/data /app/logs

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "luma.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 2: Create docker-compose.yml

```yaml
version: '3.8'

services:
  luma:
    build: .
    container_name: luma-app
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - DATABASE_URL=sqlite:////app/data/luma.db
      - LOG_LEVEL=INFO
      - ENVIRONMENT=production
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Step 3: Build and Run

```bash
# Build image
docker-compose build

# Start container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down
```

## Post-Deployment Verification

### Health Checks

```bash
# Check application health
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","version":"0.1.0"}

# Check root endpoint
curl http://localhost:8000/

# Expected response:
# {"message":"Luma is alive"}
```

### API Testing

```bash
# Create a memory
curl -X POST http://localhost:8000/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"content":"Test memory","metadata":{"source":"deployment-test"}}'

# List memories
curl http://localhost:8000/api/v1/memories

# Check API documentation
# Open browser: http://localhost:8000/docs
```

### Service Status

```bash
# Check systemd service
sudo systemctl status luma

# Check Nginx
sudo systemctl status nginx

# Check logs
sudo journalctl -u luma -n 50
tail -f /var/log/nginx/luma_access.log
```

## Monitoring and Maintenance

### Log Management

```bash
# Application logs (systemd)
sudo journalctl -u luma -f

# Nginx access logs
tail -f /var/log/nginx/luma_access.log

# Nginx error logs
tail -f /var/log/nginx/luma_error.log

# Application log file (if configured)
tail -f /home/luma/luma-app/logs/luma.log
```

### Database Maintenance

```bash
# Check database size
du -h /home/luma/luma-app/data/luma.db

# Backup database (see BACKUP_STRATEGY.md)
sqlite3 /home/luma/luma-app/data/luma.db ".backup /path/to/backup.db"

# Vacuum database (optimize)
sqlite3 /home/luma/luma-app/data/luma.db "VACUUM;"
```

### Updates and Upgrades

```bash
# Switch to luma user
sudo su - luma
cd /home/luma/luma-app

# Pull latest changes
git pull origin main

# Activate virtual environment
source venv/bin/activate

# Update dependencies
pip install --upgrade -r requirements.txt

# Exit luma user
exit

# Restart service
sudo systemctl restart luma

# Verify
sudo systemctl status luma
```

### Performance Monitoring

```bash
# Check resource usage
htop

# Check disk usage
df -h

# Check database connections (if using PostgreSQL in future)
# For SQLite, check file locks:
lsof /home/luma/luma-app/data/luma.db
```

## Troubleshooting

### Application Won't Start

```bash
# Check service status
sudo systemctl status luma

# Check logs for errors
sudo journalctl -u luma -n 100

# Common issues:
# 1. Port already in use
sudo lsof -i :8000

# 2. Permission issues
ls -la /home/luma/luma-app/data/

# 3. Configuration errors
sudo su - luma
cd /home/luma/luma-app
source venv/bin/activate
python -c "from luma.config import settings; print(settings)"
```

### Database Errors

```bash
# Check database file exists
ls -la /home/luma/luma-app/data/luma.db

# Check permissions
# Should be: -rw------- (600) owned by luma:luma

# Recreate database (WARNING: deletes all data)
rm /home/luma/luma-app/data/luma.db
sudo systemctl restart luma
```

### Nginx Issues

```bash
# Test Nginx configuration
sudo nginx -t

# Check Nginx logs
sudo tail -f /var/log/nginx/error.log

# Restart Nginx
sudo systemctl restart nginx
```

### SSL Certificate Issues

```bash
# Check certificate status
sudo certbot certificates

# Renew certificate manually
sudo certbot renew

# Test renewal
sudo certbot renew --dry-run
```

### High Memory Usage

```bash
# Check memory usage
free -h

# Reduce Uvicorn workers in systemd service
sudo nano /etc/systemd/system/luma.service
# Change: --workers 4 to --workers 2

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart luma
```

## Rollback Procedure

If deployment fails or issues arise:

```bash
# Stop service
sudo systemctl stop luma

# Switch to luma user
sudo su - luma
cd /home/luma/luma-app

# Revert to previous version
git log --oneline  # Find previous commit
git checkout <previous-commit-hash>

# Restore database backup (if needed)
cp /path/to/backup.db data/luma.db
chmod 600 data/luma.db

# Exit and restart
exit
sudo systemctl start luma
```

## Security Checklist

Before going live, verify:

- [ ] HTTPS enabled with valid SSL certificate
- [ ] Firewall configured (only necessary ports open)
- [ ] Application runs as non-root user
- [ ] Database file permissions restricted (600)
- [ ] Environment variables secured (not in version control)
- [ ] CORS restricted to trusted origins
- [ ] Debug mode disabled (ENVIRONMENT=production)
- [ ] Nginx security headers configured
- [ ] Automatic SSL renewal configured
- [ ] Backup strategy implemented
- [ ] Monitoring and alerting set up

## Additional Resources

- [Security Considerations](SECURITY.md)
- [Backup Strategy](BACKUP_STRATEGY.md)
- [Architecture Documentation](ARCHITECTURE.md)
- [API Documentation](API_DOCUMENTATION.md)

## Support

For deployment issues or questions:
- Check logs first: `sudo journalctl -u luma -n 100`
- Review troubleshooting section above
- Consult project documentation
- File an issue in the project repository

---

**Last Updated**: 2026-02-15
**Version**: 0.1.0

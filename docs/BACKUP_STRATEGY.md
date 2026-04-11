# Backup Strategy for Luma AI System

## Overview

This document outlines the backup and recovery strategy for the Luma AI system's SQLite database. A robust backup strategy is essential for data protection, disaster recovery, and business continuity.

## Table of Contents

1. [Backup Requirements](#backup-requirements)
2. [Backup Methods](#backup-methods)
3. [Backup Schedule](#backup-schedule)
4. [Backup Storage](#backup-storage)
5. [Restoration Procedures](#restoration-procedures)
6. [Automated Backup Scripts](#automated-backup-scripts)
7. [Testing and Verification](#testing-and-verification)
8. [Monitoring and Alerts](#monitoring-and-alerts)
9. [Disaster Recovery Plan](#disaster-recovery-plan)

## Backup Requirements

### Data to Backup

1. **Primary Database**: `luma.db` (SQLite database file)
2. **Configuration**: `.env` file (contains environment-specific settings)
3. **Encryption Keys**: `keys/encryption.key` (if using encryption)
4. **Application Logs**: `logs/` directory (optional, for audit trail)

### Backup Objectives

- **Recovery Point Objective (RPO)**: Maximum 24 hours of data loss
- **Recovery Time Objective (RTO)**: Restore within 1 hour
- **Retention Period**: 
  - Daily backups: 7 days
  - Weekly backups: 4 weeks
  - Monthly backups: 12 months

### Backup Types

1. **Full Backup**: Complete copy of database file
2. **Incremental Backup**: Not applicable for SQLite (use full backups)
3. **Hot Backup**: Backup while application is running
4. **Cold Backup**: Backup while application is stopped (more reliable)

## Backup Methods

### Method 1: SQLite .backup Command (Recommended)

The SQLite `.backup` command creates a consistent backup even while the database is in use.

**Advantages**:
- Works while application is running (hot backup)
- Creates consistent snapshot
- Handles locked databases gracefully
- Built-in to SQLite

**Command**:
```bash
sqlite3 /path/to/luma.db ".backup /path/to/backup/luma_backup_$(date +%Y%m%d_%H%M%S).db"
```

**Example Script**:
```bash
#!/bin/bash
# backup_sqlite.sh

DB_PATH="/home/luma/luma-app/data/luma.db"
BACKUP_DIR="/home/luma/luma-app/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/luma_backup_$TIMESTAMP.db"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Perform backup
sqlite3 "$DB_PATH" ".backup '$BACKUP_FILE'"

# Verify backup
if [ $? -eq 0 ]; then
    echo "Backup successful: $BACKUP_FILE"
    
    # Compress backup to save space
    gzip "$BACKUP_FILE"
    echo "Backup compressed: $BACKUP_FILE.gz"
else
    echo "Backup failed!" >&2
    exit 1
fi
```

### Method 2: File Copy (Simple but Less Safe)

Simple file copy works but may create inconsistent backups if database is being written to.

**Advantages**:
- Simple and fast
- No special tools required

**Disadvantages**:
- May create inconsistent backup if database is active
- Risk of corruption if database is being written

**Command**:
```bash
# Stop application first (cold backup)
sudo systemctl stop luma

# Copy database file
cp /home/luma/luma-app/data/luma.db /home/luma/luma-app/backups/luma_backup_$(date +%Y%m%d_%H%M%S).db

# Restart application
sudo systemctl start luma
```

**Use case**: Only for cold backups or when application is stopped.

### Method 3: SQLite VACUUM INTO (SQLite 3.27.0+)

Creates a compacted backup with optimized database structure.

**Advantages**:
- Creates optimized backup (removes fragmentation)
- Consistent snapshot
- Smaller file size

**Command**:
```bash
sqlite3 /path/to/luma.db "VACUUM INTO '/path/to/backup/luma_backup_$(date +%Y%m%d_%H%M%S).db'"
```

### Method 4: rsync for Remote Backups

Use rsync to copy backups to remote server for off-site storage.

**Command**:
```bash
rsync -avz --delete /home/luma/luma-app/backups/ user@backup-server:/backups/luma/
```

## Backup Schedule

### Recommended Schedule

| Frequency | Time | Retention | Method |
|-----------|------|-----------|--------|
| Hourly | Every hour | 24 hours | SQLite .backup |
| Daily | 2:00 AM | 7 days | SQLite .backup + compress |
| Weekly | Sunday 3:00 AM | 4 weeks | SQLite .backup + compress |
| Monthly | 1st of month 4:00 AM | 12 months | SQLite .backup + compress |

### Cron Configuration

```bash
# Edit crontab
crontab -e

# Add backup jobs
# Hourly backup (keep last 24)
0 * * * * /home/luma/luma-app/scripts/backup_hourly.sh

# Daily backup at 2 AM
0 2 * * * /home/luma/luma-app/scripts/backup_daily.sh

# Weekly backup on Sunday at 3 AM
0 3 * * 0 /home/luma/luma-app/scripts/backup_weekly.sh

# Monthly backup on 1st at 4 AM
0 4 1 * * /home/luma/luma-app/scripts/backup_monthly.sh

# Cleanup old backups daily at 5 AM
0 5 * * * /home/luma/luma-app/scripts/cleanup_old_backups.sh
```

## Backup Storage

### Local Storage

**Location**: `/home/luma/luma-app/backups/`

**Structure**:
```
backups/
├── hourly/
│   ├── luma_backup_20260215_010000.db.gz
│   ├── luma_backup_20260215_020000.db.gz
│   └── ...
├── daily/
│   ├── luma_backup_20260215.db.gz
│   ├── luma_backup_20260214.db.gz
│   └── ...
├── weekly/
│   ├── luma_backup_week_07_2026.db.gz
│   └── ...
└── monthly/
    ├── luma_backup_2026_02.db.gz
    └── ...
```

**Permissions**:
```bash
chmod 700 /home/luma/luma-app/backups
chmod 600 /home/luma/luma-app/backups/*.db.gz
```

### Remote/Off-Site Storage

**Options**:
1. **Remote Server**: rsync to dedicated backup server
2. **Cloud Storage**: AWS S3, Google Cloud Storage, Backblaze B2
3. **Network Attached Storage (NAS)**: Synology, QNAP
4. **External Drive**: USB drive for physical off-site storage

**Example: AWS S3 Backup**:
```bash
#!/bin/bash
# backup_to_s3.sh

BACKUP_FILE="/home/luma/luma-app/backups/daily/luma_backup_$(date +%Y%m%d).db.gz"
S3_BUCKET="s3://your-backup-bucket/luma/"

# Upload to S3
aws s3 cp "$BACKUP_FILE" "$S3_BUCKET" --storage-class STANDARD_IA

# Verify upload
if [ $? -eq 0 ]; then
    echo "Backup uploaded to S3: $BACKUP_FILE"
else
    echo "S3 upload failed!" >&2
    exit 1
fi
```

### Encryption for Off-Site Backups

Encrypt backups before uploading to untrusted storage:

```bash
#!/bin/bash
# encrypt_backup.sh

BACKUP_FILE="$1"
ENCRYPTED_FILE="$BACKUP_FILE.enc"
ENCRYPTION_KEY="/home/luma/luma-app/keys/backup_encryption.key"

# Encrypt using OpenSSL
openssl enc -aes-256-cbc -salt -pbkdf2 -in "$BACKUP_FILE" -out "$ENCRYPTED_FILE" -pass file:"$ENCRYPTION_KEY"

echo "Backup encrypted: $ENCRYPTED_FILE"
```

## Restoration Procedures

### Full Database Restoration

#### Method 1: Direct Replacement (Application Stopped)

```bash
# 1. Stop application
sudo systemctl stop luma

# 2. Backup current database (just in case)
cp /home/luma/luma-app/data/luma.db /home/luma/luma-app/data/luma.db.before_restore

# 3. Decompress backup if needed
gunzip -c /home/luma/luma-app/backups/daily/luma_backup_20260215.db.gz > /tmp/luma_restore.db

# 4. Replace database file
cp /tmp/luma_restore.db /home/luma/luma-app/data/luma.db

# 5. Set proper permissions
chmod 600 /home/luma/luma-app/data/luma.db
chown luma:luma /home/luma/luma-app/data/luma.db

# 6. Verify database integrity
sqlite3 /home/luma/luma-app/data/luma.db "PRAGMA integrity_check;"

# 7. Start application
sudo systemctl start luma

# 8. Verify application is working
curl http://localhost:8000/health

# 9. Clean up
rm /tmp/luma_restore.db
```

#### Method 2: SQLite Restore (Application Running)

```bash
# 1. Decompress backup
gunzip -c /home/luma/luma-app/backups/daily/luma_backup_20260215.db.gz > /tmp/luma_restore.db

# 2. Verify backup integrity
sqlite3 /tmp/luma_restore.db "PRAGMA integrity_check;"

# 3. Stop application
sudo systemctl stop luma

# 4. Restore using SQLite
sqlite3 /home/luma/luma-app/data/luma.db ".restore /tmp/luma_restore.db"

# 5. Start application
sudo systemctl start luma

# 6. Clean up
rm /tmp/luma_restore.db
```

### Partial Data Recovery

If you need to recover specific data without full restoration:

```bash
# 1. Open backup database
sqlite3 /tmp/luma_restore.db

# 2. Export specific data
.mode csv
.output /tmp/recovered_data.csv
SELECT * FROM memories WHERE created_at > '2026-02-01';
.quit

# 3. Import into production database
sqlite3 /home/luma/luma-app/data/luma.db
.mode csv
.import /tmp/recovered_data.csv memories
.quit
```

### Decrypting Encrypted Backups

```bash
# Decrypt backup
openssl enc -aes-256-cbc -d -pbkdf2 -in backup.db.gz.enc -out backup.db.gz -pass file:/path/to/encryption.key

# Then proceed with normal restoration
```

## Automated Backup Scripts

### Complete Backup Script

```bash
#!/bin/bash
# /home/luma/luma-app/scripts/backup_daily.sh

set -e  # Exit on error

# Configuration
DB_PATH="/home/luma/luma-app/data/luma.db"
BACKUP_DIR="/home/luma/luma-app/backups/daily"
LOG_FILE="/home/luma/luma-app/logs/backup.log"
RETENTION_DAYS=7
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATE=$(date +%Y%m%d)
BACKUP_FILE="$BACKUP_DIR/luma_backup_$DATE.db"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Create backup directory
mkdir -p "$BACKUP_DIR"

log "Starting backup: $BACKUP_FILE"

# Perform backup
if sqlite3 "$DB_PATH" ".backup '$BACKUP_FILE'"; then
    log "Backup successful: $BACKUP_FILE"
    
    # Get file size
    SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    log "Backup size: $SIZE"
    
    # Compress backup
    if gzip -f "$BACKUP_FILE"; then
        log "Backup compressed: $BACKUP_FILE.gz"
        COMPRESSED_SIZE=$(du -h "$BACKUP_FILE.gz" | cut -f1)
        log "Compressed size: $COMPRESSED_SIZE"
    else
        log "ERROR: Compression failed"
        exit 1
    fi
    
    # Verify backup integrity
    if gunzip -c "$BACKUP_FILE.gz" | sqlite3 /dev/null "PRAGMA integrity_check;" > /dev/null 2>&1; then
        log "Backup integrity verified"
    else
        log "ERROR: Backup integrity check failed"
        exit 1
    fi
    
    # Cleanup old backups
    log "Cleaning up backups older than $RETENTION_DAYS days"
    find "$BACKUP_DIR" -name "luma_backup_*.db.gz" -mtime +$RETENTION_DAYS -delete
    
    # Count remaining backups
    BACKUP_COUNT=$(ls -1 "$BACKUP_DIR"/luma_backup_*.db.gz 2>/dev/null | wc -l)
    log "Total backups retained: $BACKUP_COUNT"
    
    log "Backup completed successfully"
    
else
    log "ERROR: Backup failed"
    exit 1
fi
```

### Cleanup Script

```bash
#!/bin/bash
# /home/luma/luma-app/scripts/cleanup_old_backups.sh

# Configuration
BACKUP_BASE="/home/luma/luma-app/backups"
LOG_FILE="/home/luma/luma-app/logs/backup.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting backup cleanup"

# Cleanup hourly backups (keep 24 hours)
find "$BACKUP_BASE/hourly" -name "*.db.gz" -mtime +1 -delete
log "Hourly backups cleaned (>24 hours)"

# Cleanup daily backups (keep 7 days)
find "$BACKUP_BASE/daily" -name "*.db.gz" -mtime +7 -delete
log "Daily backups cleaned (>7 days)"

# Cleanup weekly backups (keep 4 weeks)
find "$BACKUP_BASE/weekly" -name "*.db.gz" -mtime +28 -delete
log "Weekly backups cleaned (>28 days)"

# Cleanup monthly backups (keep 12 months)
find "$BACKUP_BASE/monthly" -name "*.db.gz" -mtime +365 -delete
log "Monthly backups cleaned (>365 days)"

# Report disk usage
DISK_USAGE=$(du -sh "$BACKUP_BASE" | cut -f1)
log "Total backup disk usage: $DISK_USAGE"

log "Backup cleanup completed"
```

### Make Scripts Executable

```bash
chmod +x /home/luma/luma-app/scripts/backup_daily.sh
chmod +x /home/luma/luma-app/scripts/cleanup_old_backups.sh
```

## Testing and Verification

### Regular Backup Testing

Test backup restoration monthly to ensure backups are valid:

```bash
#!/bin/bash
# test_backup_restore.sh

TEST_DIR="/tmp/luma_backup_test"
BACKUP_FILE="/home/luma/luma-app/backups/daily/luma_backup_$(date +%Y%m%d).db.gz"

# Create test directory
mkdir -p "$TEST_DIR"

# Decompress backup
gunzip -c "$BACKUP_FILE" > "$TEST_DIR/test.db"

# Verify integrity
if sqlite3 "$TEST_DIR/test.db" "PRAGMA integrity_check;" | grep -q "ok"; then
    echo "✓ Backup integrity check passed"
else
    echo "✗ Backup integrity check FAILED"
    exit 1
fi

# Verify data
RECORD_COUNT=$(sqlite3 "$TEST_DIR/test.db" "SELECT COUNT(*) FROM memories;")
echo "✓ Backup contains $RECORD_COUNT records"

# Cleanup
rm -rf "$TEST_DIR"

echo "✓ Backup test completed successfully"
```

### Backup Verification Checklist

- [ ] Backup file exists and is not empty
- [ ] Backup file is readable
- [ ] Database integrity check passes
- [ ] Backup can be decompressed successfully
- [ ] Backup contains expected number of records
- [ ] Backup file permissions are correct (600)
- [ ] Backup is stored in correct location
- [ ] Off-site backup copy exists (if configured)

## Monitoring and Alerts

### Backup Monitoring

Monitor backup success/failure and send alerts:

```bash
#!/bin/bash
# monitor_backups.sh

BACKUP_DIR="/home/luma/luma-app/backups/daily"
ALERT_EMAIL="admin@example.com"
MAX_AGE_HOURS=26  # Alert if no backup in 26 hours

# Find most recent backup
LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/luma_backup_*.db.gz 2>/dev/null | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "ERROR: No backups found!" | mail -s "Luma Backup Alert: No Backups" "$ALERT_EMAIL"
    exit 1
fi

# Check backup age
BACKUP_AGE_HOURS=$(( ($(date +%s) - $(stat -c %Y "$LATEST_BACKUP")) / 3600 ))

if [ $BACKUP_AGE_HOURS -gt $MAX_AGE_HOURS ]; then
    echo "ERROR: Latest backup is $BACKUP_AGE_HOURS hours old" | mail -s "Luma Backup Alert: Stale Backup" "$ALERT_EMAIL"
    exit 1
fi

echo "✓ Backup monitoring: Latest backup is $BACKUP_AGE_HOURS hours old"
```

### Disk Space Monitoring

```bash
#!/bin/bash
# monitor_disk_space.sh

BACKUP_DIR="/home/luma/luma-app/backups"
THRESHOLD_PERCENT=80
ALERT_EMAIL="admin@example.com"

# Check disk usage
USAGE=$(df "$BACKUP_DIR" | tail -1 | awk '{print $5}' | sed 's/%//')

if [ $USAGE -gt $THRESHOLD_PERCENT ]; then
    echo "WARNING: Backup disk usage is ${USAGE}%" | mail -s "Luma Backup Alert: High Disk Usage" "$ALERT_EMAIL"
fi
```

## Disaster Recovery Plan

### Scenario 1: Database Corruption

**Symptoms**: Application errors, database integrity check fails

**Recovery Steps**:
1. Stop application immediately
2. Backup corrupted database for analysis
3. Restore from most recent backup
4. Verify restoration
5. Restart application
6. Investigate root cause

### Scenario 2: Accidental Data Deletion

**Symptoms**: User reports missing data

**Recovery Steps**:
1. Identify time range of deleted data
2. Find backup from before deletion
3. Extract deleted records from backup
4. Import records into production database
5. Verify data integrity

### Scenario 3: Complete System Failure

**Symptoms**: Server crash, disk failure, ransomware

**Recovery Steps**:
1. Provision new server
2. Install Luma application (see DEPLOYMENT.md)
3. Restore database from off-site backup
4. Restore configuration files
5. Verify application functionality
6. Update DNS/routing if needed

### Scenario 4: Backup System Failure

**Symptoms**: Backups not running, backup directory full

**Recovery Steps**:
1. Investigate backup failure (check logs)
2. Free up disk space if needed
3. Manually create backup immediately
4. Fix backup automation
5. Verify backups resume normally

## Best Practices

1. **3-2-1 Rule**: 
   - 3 copies of data (production + 2 backups)
   - 2 different storage types (local + cloud)
   - 1 off-site copy

2. **Test Regularly**: Test restoration monthly

3. **Automate Everything**: Use cron for automated backups

4. **Monitor Backups**: Set up alerts for backup failures

5. **Encrypt Off-Site**: Always encrypt backups stored off-site

6. **Document Procedures**: Keep this document updated

7. **Version Control**: Track backup script changes in git

8. **Secure Storage**: Restrict backup file permissions (600)

9. **Verify Integrity**: Always check database integrity after backup

10. **Plan for Disasters**: Have documented recovery procedures

## Additional Resources

- [SQLite Backup Documentation](https://www.sqlite.org/backup.html)
- [Deployment Guide](DEPLOYMENT.md)
- [Security Considerations](SECURITY.md)

---

**Last Updated**: 2026-02-15
**Version**: 0.1.0

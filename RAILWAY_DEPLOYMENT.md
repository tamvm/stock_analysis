# Railway Deployment Guide

This guide will help you deploy your Stock Analysis Dashboard to Railway.

## Prerequisites

1. **Railway CLI**: Install the Railway CLI
   ```bash
   npm install -g @railway/cli
   # OR
   curl -fsSL https://railway.app/install.sh | sh
   ```

2. **Railway Account**: Sign up at [railway.app](https://railway.app)

## Quick Deployment

### Option 1: Using the Deployment Script (Recommended)

1. Make the script executable (already done):
   ```bash
   chmod +x deploy.sh
   ```

2. Run the deployment script:
   ```bash
   ./deploy.sh
   ```

The script will guide you through the entire process!

### Option 2: Manual Deployment

1. **Login to Railway**:
   ```bash
   railway login
   ```

2. **Initialize or Link Project**:

   For new project:
   ```bash
   railway init
   ```

   For existing project:
   ```bash
   railway link
   ```

3. **Deploy**:
   ```bash
   railway up
   ```

## Configuration Files Explained

### `railway.toml`
- Configures Railway deployment settings
- Specifies the start command for Streamlit
- Sets up environment variables structure

### `Procfile`
- Alternative configuration method
- Contains the web process command

### Updated `requirements.txt`
- Removed `sqlite3` (built into Python)
- Contains all necessary dependencies

## Environment Variables

If your application requires environment variables, set them in the Railway dashboard:

1. Go to your Railway project dashboard
2. Navigate to the "Variables" tab
3. Add any required variables such as:
   - `DATABASE_URL` (if using external database)
   - API keys
   - Configuration settings

## Post-Deployment

After successful deployment:

1. **Check Status**:
   ```bash
   railway status
   ```

2. **View Logs**:
   ```bash
   railway logs
   ```

3. **Open in Browser**:
   ```bash
   railway open
   ```

## Troubleshooting

### Common Issues:

1. **Port Issues**: Railway automatically provides `$PORT` environment variable
2. **Dependencies**: Ensure all dependencies are in `requirements.txt`
3. **File Paths**: Use relative paths in your code
4. **Database**: SQLite files may not persist between deployments - consider using Railway's PostgreSQL add-on for production

### Useful Commands:

```bash
# View project info
railway status

# View real-time logs
railway logs --follow

# Connect to project shell
railway shell

# Set environment variables
railway variables set KEY=value

# Remove project link
railway unlink
```

## Data Persistence

**Important**: SQLite database files may not persist between deployments on Railway. For production use, consider:

1. **Railway PostgreSQL**: Add PostgreSQL service in Railway dashboard
2. **External Database**: Use managed database services
3. **File Storage**: Use Railway volumes or external file storage

## Next Steps

1. Set up custom domain (Railway Pro feature)
2. Configure monitoring and alerts
3. Set up CI/CD with GitHub integration
4. Consider database migration for production data
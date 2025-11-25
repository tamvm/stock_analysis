#!/bin/bash

# Railway Deployment Script for Stock Analysis Dashboard

echo "🚄 Railway Deployment Script for Stock Analysis Dashboard"
echo "==========================================================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI is not installed. Please install it first:"
    echo "   npm install -g @railway/cli"
    echo "   or"
    echo "   curl -fsSL https://railway.app/install.sh | sh"
    exit 1
fi

echo "✅ Railway CLI found"

# Check if user is logged in
if ! railway whoami &> /dev/null; then
    echo "🔐 Please log in to Railway:"
    railway login
fi

echo "✅ Railway authentication verified"

# Check if we're in a Railway project
if ! railway status &> /dev/null; then
    echo "🆕 No Railway project detected. Creating new project..."
    railway login

    # Let user choose to link existing project or create new one
    echo "Would you like to:"
    echo "1) Link to existing Railway project"
    echo "2) Create new Railway project"
    read -p "Choose option (1 or 2): " choice

    case $choice in
        1)
            echo "📎 Linking to existing project..."
            railway link
            ;;
        2)
            echo "🆕 Creating new project..."
            railway init
            ;;
        *)
            echo "❌ Invalid choice. Exiting."
            exit 1
            ;;
    esac
else
    echo "✅ Railway project detected"
fi

# Add environment variables reminder
echo ""
echo "📋 Environment Variables Checklist:"
echo "   Make sure you've set any required environment variables in Railway dashboard:"
echo "   - Database connection strings (if using external DB)"
echo "   - API keys (if any)"
echo "   - Other configuration variables"
echo ""

# Deploy to Railway
echo "🚀 Deploying to Railway..."
railway up

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Deployment successful!"
    echo "🌐 Your app should be available at the Railway-provided URL"
    echo ""
    echo "📊 View deployment status:"
    echo "   railway status"
    echo ""
    echo "📋 View logs:"
    echo "   railway logs"
    echo ""
    echo "🌍 Open in browser:"
    echo "   railway open"
else
    echo ""
    echo "❌ Deployment failed. Check the logs for more details:"
    echo "   railway logs"
    exit 1
fi
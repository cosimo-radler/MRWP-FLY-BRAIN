# Deploying the Dashboard to Vercel

This guide explains how to deploy the dashboard to Vercel.

## Prerequisites

- A GitHub account
- A Vercel account (you can sign up with your GitHub account)

## Deployment Steps

1. **Push your code to GitHub**

   Make sure your code is in a GitHub repository:
   ```
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push
   ```

2. **Connect to Vercel**

   - Go to [Vercel](https://vercel.com/) and sign in with GitHub
   - Click "New Project"
   - Import your GitHub repository
   - Keep the default settings (Vercel will auto-detect the Python project)
   - Click "Deploy"

3. **Environment Setup (if needed)**

   If your data files are not included in the repository, you'll need to set up:
   - Environment variables for any API keys
   - Add data files to the repository or use a data storage service

## How It Works

The project is structured for Vercel serverless functions:

- `api/index.py` - The entry point for the Dash application
- `vercel.json` - Configuration telling Vercel how to handle requests
- `public/` - Static assets (landing page)
- `src/visualization/dashboard.py` - The actual Dash app

## Testing Locally

You can test your Vercel deployment locally:

1. Install Vercel CLI:
   ```
   npm install -g vercel
   ```

2. Run the development server:
   ```
   vercel dev
   ```

## Important Notes

- The caching system uses a simple in-memory cache that won't persist between serverless function calls
- For large datasets, consider using a CDN or external storage service
- The initial loading time might be longer than local deployment due to the serverless cold start

## Troubleshooting

If you encounter issues:

1. Check the Vercel deployment logs
2. Ensure all dependencies are correctly specified in `api/requirements.txt`
3. Verify that file paths work correctly in a serverless environment
4. For persistent data storage issues, consider using a database or storage service 
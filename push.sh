#!/usr/bin/env bash

# Usage:
#   ./push.sh "update message"

MSG=${1:-"update"}

git init
git add -A
git commit -m "$MSG"

# Replace YOUR-REPO-URL with your actual GitHub repo, e.g.:
# git remote add origin https://github.com/username/MICRO-ASXR.git

echo "Now add your GitHub origin remote:"
echo "  git remote add origin <YOUR-REPO-URL>"
echo "Then push:"
echo "  git push -u origin main"

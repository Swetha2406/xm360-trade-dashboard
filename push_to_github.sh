#!/bin/bash
# ═══════════════════════════════════════════════════
#  Push XM360 Dashboard to YOUR GitHub account
#  Run this ONCE after creating a GitHub repo
# ═══════════════════════════════════════════════════
#
# STEP 1: Go to https://github.com/new
#         Create a repo called:  xm360-trade-dashboard
#         Leave it EMPTY (no README, no .gitignore)
#
# STEP 2: Replace YOUR_GITHUB_USERNAME below
#
# STEP 3: Run:  bash push_to_github.sh
# ═══════════════════════════════════════════════════

GITHUB_USERNAME="YOUR_GITHUB_USERNAME"
REPO_NAME="xm360-trade-dashboard"

echo ""
echo "Pushing to: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo ""

git remote add origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
git branch -M main
git push -u origin main

echo ""
echo "Done! Your repo is live at:"
echo "https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo ""
echo "Anyone can now clone it with:"
echo "git clone https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"

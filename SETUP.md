# 🧠 Machine Learning Project — Git & GitHub Setup

This document explains how to initialize a Git repository for your machine learning project, connect it to GitHub, and push your code from the terminal.

---

## 🔧 Prerequisites

Before you begin, make sure:

- `git` is installed → check with: `git --version`
- You have a [GitHub account](https://github.com)
- Authentication is set up:
  - Either via HTTPS (with Personal Access Token)
  - Or via SSH (with your public key added to GitHub)

---

## 🚀 Step-by-Step Instructions

### 1. Navigate to Your Project Folder
```bash
cd path/to/your-ml-project
```

### 2. Initialize a Git Repository
```
git init
```

### 3. Add Files and Make Your First Commit
```
git add .
git commit -m "Initial commit"
```

### 4. Create a GitHub Repository
- Go to https://github.com/new
- Choose a repository name (e.g., ml-project-2025)
- Don’t check "Initialize with README" if your local folder already has one
- Click "Create repository"

### 5. Connect Your Local Repo to GitHub
```
git remote add origin https://github.com/YOUR_USERNAME/ml-project-2025.git

```

### 6. Push Your Code to GitHub
```
git push -u origin master
```

### 🛠️ Optional: Add a .gitignore for Python

### 🔁 Daily Git Workflow
```
git add .
git commit -m "Day X: What you did"
git push
```

## Contributors

### 1. Clone the Repository
```
git clone https://github.com/bogwee/intro_ml_2025.git
```

### 2. Set Up Git Locally
```
git config --global user.name "Their Name"
git config --global user.email "their.email@example.com"
```

### 3. Best Practices for Collaboration
```
git pull origin main
```

### 4. Start Collaborating
```
git add .
git commit -m "Your message"
git push origin main
```
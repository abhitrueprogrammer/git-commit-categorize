# Git Commit Categorizer & ML.NET Analyser

**Ever stared at a terminal wondering if your commit should be `feat:`, `fix:`, or `chore:`? Stop guessing.**

This tool acts as your personalized, AI-powered commit assistant. Rather than forcing you strictly into generic conventional commits, it fetches *your* organization's actual historical commit data, uses **Machine Learning (ML.NET)** to find natural patterns(using K-means clustering) in how your team works, and leverages **Google Gemini AI** to automatically assign human-readable labels to those patterns. Finally, it provides an interactive prompt to correctly format your new commits on the fly!

## Why It's Awesome
- **Fully Autonomous ML Pipeline**: Automatically scales to your data. It uses Grid Search evaluating the Davies-Bouldin index to dynamically find the optimal number of categories (clusters) for your specific repositories. It even algorithmically penalizes outliers to naturally lean towards a readable 4-8 tag cluster size.
- **Smart & Configurable AI Labeling**: Uses the latest `gemini-2.5-flash` model algorithm to semantically label clusters (e.g., "Dependency Updates", "UI Fixes"). Features 4 configurable execution modes (`Hybrid`, `SinglePrompt`, `PerCluster`, `LocalOnly`) to tightly control API quotas and fallback natively on smart local heuristic algorithms!
- **State-of-the-Art Reliability**: Includes automatic cryptographic hashing (`SHA256`) of your datasets. If the data hasn't changed, ML training is completely bypassed and models/labels load in milliseconds from disk. Plus, native exponential backoff protects against Gemini server rate limiting.
- **Interactive Output**: Drop seamlessly into a real-time local terminal loop to instantly predict and format your next commit message against the running AI logic.

## Prerequisites
1. [.NET 9 SDK](https://dotnet.microsoft.com/download/dotnet/9.0)
2. A **GitHub Personal Access Token** (to securely fetch organization commits)
3. A **Google Gemini API Key** (for cluster semantic naming inferencing)

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abhitrueprogrammer/git-commit-categorize.git
   cd git-commit-categorize
   ```

2. **Setup your Environment Variables:**
   Create a `.env` file in your root workspace containing the following:
   ```env
   GITHUB_TOKEN=your_github_personal_access_token_here
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

3. **Restore Packages & Build:**
   ```bash
   cd ConsoleApp2
   dotnet restore
   dotnet build
   ```

## Usage

Run the project directly via the .NET CLI:
```bash
dotnet run
```

### What happens during runtime?
1. The app will pull and cache your JSON dataset.
2. If the data is new or uncached, ML.NET prepares the data (80/20 train validation split) and extracts text features. Otherwise, it efficiently reloads the cached hash model!

![Reloading Model and Cache](image/reload_model_clustername.png)

3. A Grid Search evaluates clusters, applies math penalties to prefer $K=4-8$, identifies the best valid structure, and saves the trained ML model globally (`kmeans_model.zip`).

![Cluster Recognition](image/cluster_recognition.png)

4. The Gemini LLM (or the local heuristic fallback) connects and predicts a descriptive human-readable category for the newly structured clusters.

![Cluster Examples](image/cluster_examples.png)

5. You will enter the **Interactive Labeler**:

```text
========================================
--- Interactive Commit Formatting ---
Enter a raw commit message to see it mapped to its AI cluster label.
Type 'exit' to quit.
========================================

Enter commit message: fix padding on the login button
Formatted => UI Bug Fixes: fix padding on the login button
```

![Interactive Formatting](image/interactive_commit_formatting.png)

## Architecture Overview
- `Program.cs`: Orchestrates data loading, caching/hashing invalidation checks, ML model loading, and triggers terminal execution.
- `CommitFetcher.cs`: Handles standard Octokit GitHub authentications and remote API downloading.
- `Analyser.cs`: Contains the heavy ML.NET isolated workflow (Splitting, Featurizing text, Penalized Grid Search K-Values).
- `AiClusterLabeler.cs`: Contains the robust Gemini API pipeline, fallback heuristics, multi-mode inference limits, and resilient exponential backoff retry circuits.
- `CommitInteractiveLabeler.cs`: Manages dynamic, real-time prediction formatting mapping.

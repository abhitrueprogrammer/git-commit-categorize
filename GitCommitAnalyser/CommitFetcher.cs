using DotNetEnv;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Octokit;

namespace GitCommitAnalyser
{
    public class CommitFetcher
    {
        private const int PROCESSED_COUNT = 10; // Adjust this value as needed to control how often the intermediate results are saved

        public static async Task RunAsync(string orgName, string outputPath, int? maxRepos = null)
        {
            var github = new GitHubClient(new ProductHeaderValue("GitCommitAnalyser"));
            if (!ConfigureAuthentication(github))
            {
                return;
            }

            var repoCommits = await FetchRepoCommitsAsync(github, orgName, outputPath, maxRepos);

            Console.WriteLine($"Successfully gathered commits across {repoCommits.Count} repositories.");

        }

        public static async Task<Dictionary<string, List<CommitInfo>>> FetchRepoCommitsAsync(GitHubClient github, string orgName, string filePath, int? maxRepos = null)
        {
            var repos = await github.Repository.GetAllForOrg(orgName);
            if (maxRepos is > 0)
            {
                repos = repos.Take(maxRepos.Value).ToList();
            }
            var repoCommits = new Dictionary<string, List<CommitInfo>>();
            var processedCount = 0;

            foreach (var repo in repos)
            {
                try
                {
                    var commits = await github.Repository.Commit.GetAll(repo.Owner.Login, repo.Name);
                    var commitInfos = new List<CommitInfo>();

                    foreach (var githubCommit in commits)
                    {
                        var (title, description) = SplitCommitMessage(githubCommit.Commit.Message);
                        commitInfos.Add(new CommitInfo
                        {
                            CommitName = title,
                            CommitDescription = description
                        });
                    }

                    repoCommits[repo.Name] = commitInfos;
                    processedCount++;
                    if (processedCount % PROCESSED_COUNT == 0)
                    {
                        await WriteRepoCommitsAsync(repoCommits, filePath);
                        Console.WriteLine($"Processed {repoCommits.Count} repositories so far.");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to fetch commits for {repo.Name}: {ex.Message}");
                }
            }

            await WriteRepoCommitsAsync(repoCommits, filePath);
            Console.WriteLine($"Processed {repoCommits.Count} repositories total.");

            return repoCommits;
        }

        private static async Task WriteRepoCommitsAsync(Dictionary<string, List<CommitInfo>> repoCommits, string filePath)
        {
            var jsonOptions = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(repoCommits, jsonOptions);
            await File.WriteAllTextAsync(filePath, json);
        }

        private static bool ConfigureAuthentication(GitHubClient github)
        {
            var token = Environment.GetEnvironmentVariable("GITHUB_TOKEN");
            if (string.IsNullOrWhiteSpace(token))
            {
                Console.WriteLine("GitHub token not found. Do you want to continue?(y/n)");
                var input = Console.ReadLine();
                if (input == null || !input.Equals("y", StringComparison.OrdinalIgnoreCase))
                {
                    Console.WriteLine("Exiting application.");
                    return false;
                }

                return true;
            }

            github.Credentials = new Credentials(token);
            return true;
        }

        private static (string Title, string Description) SplitCommitMessage(string message)
        {
            if (string.IsNullOrWhiteSpace(message))
            {
                return (string.Empty, string.Empty);
            }

            var parts = message.Split(new[] { "\r\n", "\n" }, 2, StringSplitOptions.None);
            var title = parts[0].Trim();
            var description = parts.Length > 1 ? parts[1].Trim() : string.Empty;
            return (title, description);
        }
    }

    public class CommitInfo
    {
        public string CommitName { get; set; }
        public string CommitDescription { get; set; }
    }
}

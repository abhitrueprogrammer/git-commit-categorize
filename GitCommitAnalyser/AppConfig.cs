using System;

namespace GitCommitAnalyser
{
    public class AppConfig
    {
        public string OrgName { get; set; }
        public int? MaxRepos { get; set; }
        public string OutputPath { get; set; } = "repo_commits.json";
        public string KFilePath { get; set; } = "best_k.txt";
        public string LabelsFilePath { get; set; } = "cluster_labels.json";
        public string HashFilePath { get; set; } = "data_hash.txt";
        public string ModelFilePath { get; set; } = "kmeans_model.zip";

        public static AppConfig PromptUserForConfiguration()
        {
            var config = new AppConfig();

            Console.Write("Enter GitHub Organization name [default: CodeChefVIT]: ");
            var orgInput = Console.ReadLine();
            config.OrgName = string.IsNullOrWhiteSpace(orgInput) ? "CodeChefVIT" : orgInput.Trim();

            Console.Write("Enter Max Repos to fetch (leave blank for all) [default: all]: ");
            var maxReposInput = Console.ReadLine();
            if (int.TryParse(maxReposInput, out int count) && count > 0)
            {
                config.MaxRepos = count;
            }

            return config;
        }
    }
}

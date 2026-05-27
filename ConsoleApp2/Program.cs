using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using DotNetEnv;
using Microsoft.ML;

namespace GitCommitAnalyser
{
    class Program
    {
        static async Task Main(string[] args)
        {
            // cluster first then categorize
            Env.TraversePath().Load();

            var orgName = "CodeChefVIT";
            var outputPath = "repo_commits.json";
            
            //await CommitFetcher.RunAsync(orgName, outputPath);

            var mlContext = new MLContext(seed: 0);
            var dataView = LoadJsonDataForML(mlContext, outputPath);

            if (dataView != null)
            {
                Console.WriteLine($"Loaded {dataView.GetRowCount()} rows into ML.NET IDataView.");
            }
        }

        static IDataView LoadJsonDataForML(MLContext mlContext, string jsonFilePath)
        {
            if (!File.Exists(jsonFilePath))
            {
                Console.WriteLine($"File not found: {jsonFilePath}");
                return null;
            }

            var json = File.ReadAllText(jsonFilePath);
            var repoCommits = JsonSerializer.Deserialize<Dictionary<string, List<CommitInfo>>>(json);
            var flatData = new List<CommitMLData>();

            if (repoCommits != null)
            {
                foreach (var kvp in repoCommits)
                {
                    string repoName = kvp.Key;
                    if (kvp.Value == null) continue;
                    
                    foreach (var commit in kvp.Value)
                    {
                        flatData.Add(new CommitMLData
                        {
                            Repository = repoName,
                            CommitName = commit.CommitName ?? string.Empty,
                            CommitDescription = commit.CommitDescription ?? string.Empty
                        });
                    }
                }
            }

            return mlContext.Data.LoadFromEnumerable(flatData);
        }
    }

    public class CommitMLData
    {
        public string Repository { get; set; }
        public string CommitName { get; set; }
        public string CommitDescription { get; set; }
    }
}

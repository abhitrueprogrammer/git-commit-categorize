using System;
using System.IO;
using System.Threading.Tasks;
using DotNetEnv;

namespace ConsoleApp2
{
    class Program
    {
        static async Task Main(string[] args)
        {
            // cluster first then categorize
            Env.TraversePath().Load();

            var orgName = "CodeChefVIT";
            var outputPath = "repo_commits.json";
            
            await CommitFetcher.RunAsync(orgName, outputPath);
        }
    }
}

using System;
using System.IO;
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
            var kFilePath = "best_k.txt";
            
            //await CommitFetcher.RunAsync(orgName, outputPath);

            var mlContext = new MLContext(seed: 0);
            
            // 1. Load data
            var dataView = Analyser.LoadJsonDataForML(mlContext, outputPath);

            if (dataView != null)
            {
                Console.WriteLine($"Loaded {dataView.GetRowCount()} rows into ML.NET IDataView.");

                // 2. Split data into 80% train and 20% test
                var split = Analyser.SplitData(mlContext, dataView);

                // 3. Featurize Text
                var featurizer = Analyser.FeaturizeText(mlContext);

                // 4. Find Best K using Grid Search or load from file
                int bestK = Analyser.GetOrFindBestK(mlContext, split.TrainSet, featurizer, kFilePath);

                // 5. Train KMeans Clusterer
                var model = Analyser.TrainKMeansClusterer(mlContext, split.TrainSet, featurizer, bestK);

                // 6. Print 2 examples from each cluster
                Analyser.PrintClusterExamples(mlContext, dataView, model);
            }
        }
    }
}

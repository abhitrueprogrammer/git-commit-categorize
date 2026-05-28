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

            var labelsFilePath = "cluster_labels.json";

            if (dataView != null)
            {
                Console.WriteLine($"Loaded {dataView.GetRowCount()} rows into ML.NET IDataView.");

                var hashFilePath = "data_hash.txt";
                var modelFilePath = "kmeans_model.zip";

                string currentHash = Analyser.CalculateFileHash(outputPath);
                string savedHash = File.Exists(hashFilePath) ? File.ReadAllText(hashFilePath) : string.Empty;

                if (currentHash != savedHash)
                {
                    Console.WriteLine("Data file changed or hash not found. Invalidating cache (K, labels, model)...");
                    if (File.Exists(kFilePath)) File.Delete(kFilePath);
                    if (File.Exists(labelsFilePath)) File.Delete(labelsFilePath);
                    if (File.Exists(modelFilePath)) File.Delete(modelFilePath);
                    File.WriteAllText(hashFilePath, currentHash);
                }

                // 2. Split data into 80% train and 20% test
                var split = Analyser.SplitData(mlContext, dataView);

                ITransformer model;
                if (File.Exists(modelFilePath))
                {
                    Console.WriteLine($"Loading existing model from {modelFilePath}...");
                    model = mlContext.Model.Load(modelFilePath, out var schema);
                }
                else
                {
                    // 3. Featurize Text
                    var featurizer = Analyser.FeaturizeText(mlContext);

                    // 4. Find Best K using Grid Search or load from file
                    int bestK = Analyser.GetOrFindBestK(mlContext, split.TrainSet, featurizer, kFilePath);

                    // 5. Train KMeans Clusterer
                    model = Analyser.TrainKMeansClusterer(mlContext, split.TrainSet, featurizer, bestK);

                    mlContext.Model.Save(model, split.TrainSet.Schema, modelFilePath);
                    Console.WriteLine($"Saved newly trained model to {modelFilePath}.");
                }

                var clusters = Analyser.GetClusters(mlContext, split.TrainSet, model);

                // 6. Predict cluster names via Gemini (or load from cache)
                var clusterNames = await AiClusterLabeler.PredictClusterNamesAsync(clusters, labelsFilePath);

                // 7. Print examples from each cluster with readable names
                Analyser.PrintClusterExamples(clusters, clusterNames);

                // 8. Start the interactive labeling loop
                var interactiveLabeler = new CommitInteractiveLabeler(mlContext, model, clusterNames);
                interactiveLabeler.StartInteractiveLoop();
            }
        }
    }
}

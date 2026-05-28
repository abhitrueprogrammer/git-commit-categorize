using Microsoft.ML.Data;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.ML;

namespace GitCommitAnalyser
{
    public class CommitMLData
    {
        public string Repository { get; set; }
        public string CommitName { get; set; }
        public string CommitDescription { get; set; }
    }

    public class CommitPredictionWithData : CommitMLData
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId { get; set; }
        
        [ColumnName("Score")]
        public float[] Distances { get; set; }
    }


    internal class Analyser
    {
        private const string FeaturesColumnName = "Features";

        public static string CalculateFileHash(string filePath)
        {
            if (!File.Exists(filePath)) return string.Empty;
            using var sha256 = SHA256.Create();
            using var stream = File.OpenRead(filePath);
            var hash = sha256.ComputeHash(stream);
            return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
        }

        public static IDataView LoadJsonDataForML(MLContext mlContext, string jsonFilePath)
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

        public static DataOperationsCatalog.TrainTestData SplitData(MLContext mlContext, IDataView data)
        {
            return mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
        }

        public static IEstimator<ITransformer> FeaturizeText(MLContext mlContext)
        {
            return mlContext.Transforms.Text.FeaturizeText(FeaturesColumnName, nameof(CommitMLData.CommitName));
        }

        public static int GetOrFindBestK(MLContext mlContext, IDataView trainData, IEstimator<ITransformer> featurizer, string kFilePath)
        {
            if (File.Exists(kFilePath))
            {
                if (int.TryParse(File.ReadAllText(kFilePath), out int savedK))
                {
                    Console.WriteLine($"Loaded best K = {savedK} from file {kFilePath}.");
                    return savedK;
                }
            }

            Console.WriteLine("Finding best K via Grid Search using validation split...");
            var split = mlContext.Data.TrainTestSplit(trainData, testFraction: 0.2);
            var subTrainData = split.TrainSet;
            var validationData = split.TestSet;

            int bestK = 2;
            double bestMetric = double.MaxValue; // Lower Davies-Bouldin is better for measuring clustering quality

            for (int k = 2; k <= 10; k++)
            {
                var pipeline = featurizer.Append(mlContext.Clustering.Trainers.KMeans(featureColumnName: FeaturesColumnName, numberOfClusters: k));
                var model = pipeline.Fit(subTrainData);

                var predictions = model.Transform(validationData);
                var metrics = mlContext.Clustering.Evaluate(predictions, labelColumnName: null, scoreColumnName: "Score", featureColumnName: FeaturesColumnName);

                Console.WriteLine($"K = {k} | Davies-Bouldin: {metrics.DaviesBouldinIndex:F4} | Avg Distance: {metrics.AverageDistance:F4}");

                if (double.IsNaN(metrics.DaviesBouldinIndex)) continue;

                // We prioritize Davies-Bouldin index for clustering quality
                if (metrics.DaviesBouldinIndex < bestMetric)
                {
                    bestMetric = metrics.DaviesBouldinIndex;
                    bestK = k;
                }
            }

            Console.WriteLine($"Best K found: {bestK}. Saving to {kFilePath}.");
            File.WriteAllText(kFilePath, bestK.ToString());
            return bestK;
        }

        public static ITransformer TrainKMeansClusterer(MLContext mlContext, IDataView trainData, IEstimator<ITransformer> featurizer, int k)
        {
            var pipeline = featurizer.Append(mlContext.Clustering.Trainers.KMeans(featureColumnName: FeaturesColumnName, numberOfClusters: k));
            return pipeline.Fit(trainData);
        }

        public static IEnumerable<IGrouping<uint, CommitPredictionWithData>> GetClusters(MLContext mlContext, IDataView data, ITransformer model)
        {
            var predictions = model.Transform(data);
            var results = mlContext.Data.CreateEnumerable<CommitPredictionWithData>(predictions, reuseRowObject: false).ToList();
            return results.GroupBy(x => x.PredictedClusterId).OrderBy(g => g.Key);
        }

        public static void PrintClusterExamples(IEnumerable<IGrouping<uint, CommitPredictionWithData>> clusters, Dictionary<uint, string> clusterNames = null)
        {
            Console.WriteLine("\n--- Cluster Examples ---");
            foreach (var cluster in clusters)
            {
                var name = clusterNames != null && clusterNames.TryGetValue(cluster.Key, out var cn) ? cn: $"Cluster {cluster.Key}";
                Console.WriteLine($"\n{name}:");
                foreach (var example in cluster.Take(2)) // 2 examples each from each cluster
                {
                    Console.WriteLine($"  - [{example.Repository}] {example.CommitName}");
                }
            }
        }
    }
}

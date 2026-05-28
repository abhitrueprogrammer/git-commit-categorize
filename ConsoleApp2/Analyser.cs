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

        public static async Task<Dictionary<uint, string>> PredictClusterNamesAsync(IEnumerable<IGrouping<uint, CommitPredictionWithData>> clusters, string labelsFilePath, string modelName = "gemini-2.5-flash")
        {
            if (File.Exists(labelsFilePath))
            {
                var json = File.ReadAllText(labelsFilePath);
                var dict = JsonSerializer.Deserialize<Dictionary<string, string>>(json);
                if (dict != null && dict.Count > 0)
                {
                    Console.WriteLine($"Loaded cluster names from {labelsFilePath}.");
                    return dict.ToDictionary(k => uint.Parse(k.Key), v => v.Value);
                }
            }

            var apiKey = Environment.GetEnvironmentVariable("GEMINI_API_KEY");
            if (string.IsNullOrEmpty(apiKey))
            {
                Console.WriteLine("GEMINI_API_KEY environment variable not found. Using default cluster numeric names.");
                return clusters.ToDictionary(g => g.Key, g => $"Cluster {g.Key}");
            }

            var clusterNames = new System.Collections.Generic.Dictionary<uint, string>();
            using var httpClient = new System.Net.Http.HttpClient();

            Console.WriteLine("\nPredicting cluster names using Gemini API...");
            foreach (var cluster in clusters)
            {
                var commitsToUse = cluster.Take(10).Select(c => c.CommitName).ToList();
                var prompt = "Based on the following git commit messages, provide a short 1-3 word category name for this cluster.\n\n" +
                             "Commits:\n" + string.Join("\n", commitsToUse) + "\n\nCategory name:";

                var requestBody = new
                {
                    contents = new[]
                    {
                        new
                        {
                            parts = new[] { new { text = prompt } }
                        }
                    }
                };

                var url = $"https://generativelanguage.googleapis.com/v1beta/models/{modelName}:generateContent?key={apiKey}";
                var jsonContent = new System.Net.Http.StringContent(JsonSerializer.Serialize(requestBody), System.Text.Encoding.UTF8, "application/json");

                try
                {
                    var response = await httpClient.PostAsync(url, jsonContent);
                    if (response.IsSuccessStatusCode)
                    {
                        var responseString = await response.Content.ReadAsStringAsync();
                        using var doc = JsonDocument.Parse(responseString);
                        var text = doc.RootElement
                            .GetProperty("candidates")[0]
                            .GetProperty("content")
                            .GetProperty("parts")[0]
                            .GetProperty("text").GetString();

                        var cleanedName = text?.Trim().TrimEnd('\r', '\n', '.', '\"', '\'')
                            .Replace("'", ""); // Additional cleanup for single quotes
                        clusterNames[cluster.Key] = string.IsNullOrWhiteSpace(cleanedName) ? $"Cluster {cluster.Key}" : cleanedName;
                        Console.WriteLine($"Cluster {cluster.Key} predicted as: {clusterNames[cluster.Key]}");
                    }
                    else
                    {
                        Console.WriteLine($"Failed to predict name for Cluster {cluster.Key}. Status: {response.StatusCode}");
                        clusterNames[cluster.Key] = $"Cluster {cluster.Key}";
                    }
                }
                catch (System.Exception ex)
                {
                    Console.WriteLine($"Error predicting name for Cluster {cluster.Key}: {ex.Message}");
                    clusterNames[cluster.Key] = $"Cluster {cluster.Key}";
                }
            }

            // Save for future runs
            var saveFormat = clusterNames.ToDictionary(k => k.Key.ToString(), v => v.Value);
            var jsonOut = JsonSerializer.Serialize(saveFormat, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(labelsFilePath, jsonOut);

            return clusterNames;
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

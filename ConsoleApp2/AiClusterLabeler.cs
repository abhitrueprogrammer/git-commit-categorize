using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;

namespace GitCommitAnalyser
{
    public class AiClusterLabeler
    {
        public static async Task<Dictionary<uint, string>> PredictClusterNamesAsync(IEnumerable<IGrouping<uint, CommitPredictionWithData>> clusters, string labelsFilePath, string modelName = "gemini-2.5-flash", int commitProcessCount = 20)
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
                var commitsToUse = cluster.Take(commitProcessCount).Select(c => c.CommitName).ToList();
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
    }
}
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;

namespace GitCommitAnalyser
{
    public enum ClusterLabelingMode
    {
        PerCluster,
        SinglePrompt,
        LocalOnly,
        Hybrid
    }

    public class AiClusterLabeler
    {
        public static async Task<Dictionary<uint, string>> PredictClusterNamesAsync(
            IEnumerable<IGrouping<uint, CommitPredictionWithData>> clusters, 
            string labelsFilePath, 
            ClusterLabelingMode mode = ClusterLabelingMode.SinglePrompt, 
            string modelName = "gemini-2.5-flash")
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

            var clusterNames = new Dictionary<uint, string>();
            var apiKey = Environment.GetEnvironmentVariable("GEMINI_API_KEY");

            bool needsApi = mode == ClusterLabelingMode.PerCluster || mode == ClusterLabelingMode.SinglePrompt || mode == ClusterLabelingMode.Hybrid;
            
            if (needsApi && string.IsNullOrEmpty(apiKey))
            {
                Console.WriteLine("GEMINI_API_KEY not found. Falling back to LocalOnly mode.");
                mode = ClusterLabelingMode.LocalOnly;
            }

            using var httpClient = new HttpClient();
            string url = $"https://generativelanguage.googleapis.com/v1beta/models/{modelName}:generateContent?key={apiKey}";

            Console.WriteLine($"\nPredicting cluster names using {mode} mode...");

            switch (mode)
            {
                case ClusterLabelingMode.LocalOnly:
                    clusterNames = ProcessLocalOnly(clusters);
                    break;
                case ClusterLabelingMode.SinglePrompt:
                    clusterNames = await ProcessSinglePromptAsync(httpClient, url, clusters);
                    break;
                case ClusterLabelingMode.PerCluster:
                    clusterNames = await ProcessPerClusterAsync(httpClient, url, clusters);
                    break;
                case ClusterLabelingMode.Hybrid:
                    clusterNames = await ProcessHybridAsync(httpClient, url, clusters);
                    break;
            }

            // Ensure all clusters have a valid label, otherwise halt execution to prevent using bad tags
            foreach (var cluster in clusters)
            {
                if (!clusterNames.ContainsKey(cluster.Key) || string.IsNullOrWhiteSpace(clusterNames[cluster.Key]))
                {
                    Console.WriteLine($"\nCritical Error: Failed to resolve a valid label for Cluster {cluster.Key}. Stopping program to prevent using placeholder tags.");
                    Environment.Exit(1);
                }
            }

            // Save for future runs
            var saveFormat = clusterNames.ToDictionary(k => k.Key.ToString(), v => v.Value);
            var jsonOut = JsonSerializer.Serialize(saveFormat, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(labelsFilePath, jsonOut);

            return clusterNames;
        }

        private static Dictionary<uint, string> ProcessLocalOnly(IEnumerable<IGrouping<uint, CommitPredictionWithData>> clusters)
        {
            var clusterNames = new Dictionary<uint, string>();
            foreach (var cluster in clusters)
            {
                string label = GetLocalHeuristicLabel(cluster);
                if (!string.IsNullOrEmpty(label))
                {
                    clusterNames[cluster.Key] = label;
                    Console.WriteLine($"Cluster {cluster.Key} resolved locally as: {label}");
                }
            }
            return clusterNames;
        }

        private static async Task<Dictionary<uint, string>> ProcessSinglePromptAsync(HttpClient httpClient, string url, IEnumerable<IGrouping<uint, CommitPredictionWithData>> clusters)
        {
            var clusterNames = new Dictionary<uint, string>();
            if (!clusters.Any()) return clusterNames;

            var promptBuilder = new System.Text.StringBuilder();
            promptBuilder.AppendLine("Analyze the following clusters of git commit messages and provide a short 1-3 word category name for each.");
            promptBuilder.AppendLine("Return ONLY a valid JSON object mapping the cluster ID (as a string) to the category name. Example: {\"1\": \"Bug Fixes\", \"2\": \"Merges\"}");
            promptBuilder.AppendLine("\nClusters:");

            foreach (var cluster in clusters)
            {
                var commitsToUse = cluster.Take(7).Select(c => c.CommitName);
                promptBuilder.AppendLine($"Cluster {cluster.Key}:");
                foreach (var c in commitsToUse) promptBuilder.AppendLine($"- {c}");
            }

            var requestBody = new { contents = new[] { new { parts = new[] { new { text = promptBuilder.ToString() } } } } };

            string responseText = await CallGeminiWithRetryAsync(httpClient, url, requestBody);
            
            if (!string.IsNullOrEmpty(responseText))
            {
                try
                {
                    string cleanJson = responseText.Replace("```json", "").Replace("```", "").Trim();
                    var map = JsonSerializer.Deserialize<Dictionary<string, string>>(cleanJson);
                    if (map != null)
                    {
                        foreach (var kvp in map)
                        {
                            if (uint.TryParse(kvp.Key, out uint id))
                            {
                                clusterNames[id] = CleanLabel(kvp.Value);
                                Console.WriteLine($"Cluster {id} predicted as: {clusterNames[id]}");
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to parse SinglePrompt JSON response: {ex.Message}");
                }
            }
            return clusterNames;
        }

        private static async Task<Dictionary<uint, string>> ProcessPerClusterAsync(HttpClient httpClient, string url, IEnumerable<IGrouping<uint, CommitPredictionWithData>> clusters)
        {
            var clusterNames = new Dictionary<uint, string>();
            foreach (var cluster in clusters)
            {
                var commitsToUse = cluster.Take(20).Select(c => c.CommitName).ToList();
                var prompt = "Based on the following git commit messages, provide a short 1-3 word category name for this cluster.\n\n" +
                             "Return ONLY the category name.\n\n" +
                             "Guidelines:\n" +
                             "- Focus on the main action or intent shared across the commits.\n" +
                             "- Ignore noisy identifiers such as issue numbers, dependency names, hashes, usernames, and version numbers.\n" +
                             "- Prefer broad reusable engineering categories.\n\n" +
                             "Commits:\n" + string.Join("\n", commitsToUse) + "\n\nCategory:";

                var requestBody = new { contents = new[] { new { parts = new[] { new { text = prompt } } } } };

                string responseText = await CallGeminiWithRetryAsync(httpClient, url, requestBody);
                
                if (!string.IsNullOrEmpty(responseText))
                {
                    clusterNames[cluster.Key] = CleanLabel(responseText);
                    Console.WriteLine($"Cluster {cluster.Key} predicted as: {clusterNames[cluster.Key]}");
                }
                else
                {
                    Console.WriteLine($"Failed to predict name for Cluster {cluster.Key} (API fallback).");
                }
            }
            return clusterNames;
        }

        private static async Task<Dictionary<uint, string>> ProcessHybridAsync(HttpClient httpClient, string url, IEnumerable<IGrouping<uint, CommitPredictionWithData>> clusters)
        {
            var clusterNames = ProcessLocalOnly(clusters);
            var unresolved = clusters.Where(c => !clusterNames.ContainsKey(c.Key)).ToList();

            if (unresolved.Any())
            {
                Console.WriteLine($"Hybrid mode: {unresolved.Count} clusters unresolved locally. Sending to Gemini via SinglePrompt...");
                var geminiNames = await ProcessSinglePromptAsync(httpClient, url, unresolved);
                foreach (var kvp in geminiNames)
                {
                    clusterNames[kvp.Key] = kvp.Value;
                }
            }

            return clusterNames;
        }

        private static string GetLocalHeuristicLabel(IEnumerable<CommitPredictionWithData> commits)
        {
            int count = commits.Count();
            if (count == 0) return null;

            var msgs = commits.Select(c => c.CommitName.ToLowerInvariant()).ToList();
            
            if ((double)msgs.Count(m => m.Contains("merge pull request") || m.Contains("merge branch")) / count > 0.4) 
                return "Merges";

            if ((double)msgs.Count(m => m.Contains("bump") || m.Contains("dependency") || m.Contains("npm") || m.Contains("yarn")) / count > 0.4) 
                return "Dependencies";

            if ((double)msgs.Count(m => m.Contains("doc") || m.Contains("readme")) / count > 0.4) 
                return "Documentation";

            if ((double)msgs.Count(m => m.Contains("fix") || m.Contains("bug") || m.Contains("patch")) / count > 0.4) 
                return "Bug Fixes";

            return null; // Unresolved
        }

        private static async Task<string> CallGeminiWithRetryAsync(HttpClient httpClient, string url, object requestBody, int maxRetries = 3)
        {
            for (int attempt = 1; attempt <= maxRetries; attempt++)
            {
                try
                {
                    // Recreate StringContent per retry to avoid stream exhaustion limits
                    var jsonContent = new StringContent(JsonSerializer.Serialize(requestBody), System.Text.Encoding.UTF8, "application/json");
                    var response = await httpClient.PostAsync(url, jsonContent);
                    
                    if (response.IsSuccessStatusCode)
                    {
                        var responseString = await response.Content.ReadAsStringAsync();
                        using var doc = JsonDocument.Parse(responseString);
                        return doc.RootElement
                            .GetProperty("candidates")[0]
                            .GetProperty("content")
                            .GetProperty("parts")[0]
                            .GetProperty("text").GetString();
                    }
                    else
                    {
                        var errorBody = await response.Content.ReadAsStringAsync();
                        // Filter out non-retriable codes (like 400 Bad Request) but allow 429 and 5xx 
                        if ((int)response.StatusCode >= 400 && (int)response.StatusCode < 500 && response.StatusCode != System.Net.HttpStatusCode.TooManyRequests)
                        {
                            Console.WriteLine($"Client Error {response.StatusCode}. Aborting retry.");
                            return null; // Don't retry a bad malformed json request
                        }

                        Console.WriteLine($"API Status {response.StatusCode} on attempt {attempt}: {errorBody}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"API Request Error on attempt {attempt}: {ex.Message}");
                }

                if (attempt < maxRetries)
                {
                    Console.WriteLine("Waiting before retrying...");
                    await Task.Delay(2000 * attempt);
                }
            }
            
            Console.WriteLine("Max retries reached. Gracefully falling back.");
            return null;
        }

        private static string CleanLabel(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) return null;
            return text.Trim().TrimEnd('\r', '\n', '.', '\"', '\'').Replace("'", "");
        }
    }
}
using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace GitCommitAnalyser
{
    public class CommitInteractiveLabeler
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _model;
        private readonly Dictionary<uint, string> _clusterNames;
        private readonly PredictionEngine<CommitMLData, CommitPredictionWithData> _predictionEngine;

        public CommitInteractiveLabeler(MLContext mlContext, ITransformer model, Dictionary<uint, string> clusterNames)
        {
            _mlContext = mlContext;
            _model = model;
            _clusterNames = clusterNames;

            // Create a Prediction Engine for single-item inference
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<CommitMLData, CommitPredictionWithData>(_model);
        }

        public void StartInteractiveLoop()
        {
            Console.WriteLine("\n========================================");
            Console.WriteLine("--- Interactive Commit Formatting ---");
            Console.WriteLine("Enter a raw commit message to see it mapped to its AI cluster label.");
            Console.WriteLine("Type 'exit' to quit.");
            Console.WriteLine("========================================\n");

            while (true)
            {
                Console.Write("Enter commit message: ");
                var input = Console.ReadLine();

                if (string.IsNullOrWhiteSpace(input)) continue;
                if (input.Equals("exit", StringComparison.OrdinalIgnoreCase)) 
                {
                    Console.WriteLine("Exiting interactive mode.");
                    break;
                }

                var formattedCommit = FormatCommitMessage(input);
                Console.WriteLine($"Formatted => {formattedCommit}\n");
            }
        }

        public string FormatCommitMessage(string commitName)
        {
            // Prepare the input data for the ML.NET model
            var inputData = new CommitMLData 
            { 
                CommitName = commitName, 
                CommitDescription = string.Empty, 
                Repository = "InteractiveInput" 
            };

            // Predict which cluster it belongs to
            var prediction = _predictionEngine.Predict(inputData);

            // Fetch the human-readable label from the Gemini cached Dictionary
            string label = _clusterNames != null && _clusterNames.TryGetValue(prediction.PredictedClusterId, out var name) 
                ? name 
                : $"Cluster_{prediction.PredictedClusterId}";

            // Map spaces to hyphens/underscores if you want a strict prefix, 
            // but keeping it as the Gemini-provided label string works directly.
            // Result: "Bug Fix / UI: fixed the button alignment"
            return $"{label}: {commitName}";
        }
    }
}

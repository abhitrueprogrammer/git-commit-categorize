using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;

namespace GitCommitAnalyser
{
    public class AppOrchestrator
    {
        private readonly AppConfig _config;

        public AppOrchestrator(AppConfig config)
        {
            _config = config;
        }

        public async Task RunAsync()
        {
            // 1. Ensure Data Exists
            if (!File.Exists(_config.OutputPath))
            {
                Console.WriteLine($"\nFile '{_config.OutputPath}' not found. Fetching commits for {_config.OrgName} from GitHub...");
                await CommitFetcher.RunAsync(_config.OrgName, _config.OutputPath, _config.MaxRepos);
            }

            var mlContext = new MLContext(seed: 0);

            // 2. Load data
            var dataView = Analyser.LoadJsonDataForML(mlContext, _config.OutputPath);
            if (dataView == null) return;

            Console.WriteLine($"Loaded {dataView.GetRowCount()} rows into ML.NET IDataView.");

            // 3. Check Data Integrity / Caching
            ValidateCache();

            // 4. Split data into 80% train and 20% test
            var split = Analyser.SplitData(mlContext, dataView);

            // 5. ML Training or Loading
            ITransformer model;
            if (File.Exists(_config.ModelFilePath))
            {
                Console.WriteLine($"Loading existing model from {_config.ModelFilePath}...");
                model = mlContext.Model.Load(_config.ModelFilePath, out var schema);
            }
            else
            {
                var featurizer = Analyser.FeaturizeText(mlContext);
                int bestK = Analyser.GetOrFindBestK(mlContext, split.TrainSet, featurizer, _config.KFilePath);

                model = Analyser.TrainKMeansClusterer(mlContext, split.TrainSet, featurizer, bestK);

                mlContext.Model.Save(model, split.TrainSet.Schema, _config.ModelFilePath);
                Console.WriteLine($"Saved newly trained model to {_config.ModelFilePath}.");
            }

            var clusters = Analyser.GetClusters(mlContext, split.TrainSet, model);

            // 6. Predict cluster names via Gemini (or load from cache)
            var clusterNames = await AiClusterLabeler.PredictClusterNamesAsync(clusters, _config.LabelsFilePath);

            // 7. Print examples from each cluster with readable names
            Analyser.PrintClusterExamples(clusters, clusterNames);

            // 8. Start the interactive labeling loop
            var interactiveLabeler = new CommitInteractiveLabeler(mlContext, model, clusterNames);
            interactiveLabeler.StartInteractiveLoop();
        }

        private void ValidateCache()
        {
            string currentHash = Analyser.CalculateFileHash(_config.OutputPath);
            string savedHash = File.Exists(_config.HashFilePath) ? File.ReadAllText(_config.HashFilePath) : string.Empty;

            if (currentHash != savedHash)
            {
                Console.WriteLine("Data file changed or hash not found. Invalidating cache (K, labels, model)...");
                if (File.Exists(_config.KFilePath)) File.Delete(_config.KFilePath);
                if (File.Exists(_config.LabelsFilePath)) File.Delete(_config.LabelsFilePath);
                if (File.Exists(_config.ModelFilePath)) File.Delete(_config.ModelFilePath);

                File.WriteAllText(_config.HashFilePath, currentHash);
            }
        }
    }
}

using System.Threading.Tasks;
using DotNetEnv;

namespace GitCommitAnalyser
{
    class Program
    {
        static async Task Main(string[] args)
        {
            // Load environment variables (.env file)
            Env.TraversePath().Load();

            // Prompt user for input and configuration
            var config = AppConfig.PromptUserForConfiguration();

            // Run the main application orchestrator
            var orchestrator = new AppOrchestrator(config);
            await orchestrator.RunAsync();
        }
    }
}

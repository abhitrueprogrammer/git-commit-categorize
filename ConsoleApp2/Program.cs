// See https://aka.ms/new-console-template for more information


using Octokit;

var github = new GitHubClient(new ProductHeaderValue("MyAmazingApp"));
var user = await github.User.Get("abhitrueprogrammer");
Console.WriteLine(user.Followers + $" folks love {user.Login}!");
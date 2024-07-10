#pragma warning disable SKEXP0001
#pragma warning disable SKEXP0010
#pragma warning disable SKEXP0050

using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Memory;
using Microsoft.SemanticKernel.Plugins.Memory;

const string modelPhi3 = @"phi3:medium";
const string endpointOllama = "http://localhost:11434";

var kernelBuilder = Kernel.CreateBuilder()
                          .AddOpenAIChatCompletion(modelId: modelPhi3, endpoint: new Uri(endpointOllama), apiKey: null, serviceId: modelPhi3)
                          .AddLocalTextEmbeddingGeneration()
                          ;

var kernel = kernelBuilder.Build();

/* Add memory */
var embeddingGenerator = kernel.Services.GetRequiredService<ITextEmbeddingGenerationService>();
var memory = new SemanticTextMemory(new VolatileMemoryStore(), embeddingGenerator);

// Add some facts to the collection...
const string MemoryCollectionName = @"MyPersonalFacts";

await memory.SaveInformationAsync(MemoryCollectionName, id: Guid.NewGuid().ToString(), text: @"Rodrigo Liberoff holds the role of Senior Cloud and Software Architect, Technical Leader in AI and Generative Language Models at ENCAMINA. ");
await memory.SaveInformationAsync(MemoryCollectionName, id: Guid.NewGuid().ToString(), text: @"Rodrigo Liberoff is a Computer Engineering Professional (or Computing) with over 20 years of practical experience, dynamic in attitude and adaptable to change; passionate about the design, conceptualization, and development of software systems, mainly using the C# programming language and agile methodologies (such as Scrum, SAFe), the creation of software architectures, and research on new technologies and their possible dissemination in the community, my colleagues, clients, their projects, or the company, focused mainly on Microsoft technologies, especially the .NET platform (currently .NET 6, 7, and 8).");
await memory.SaveInformationAsync(MemoryCollectionName, id: Guid.NewGuid().ToString(), text: @"Rodrigo Liberoff currently works at ENCAMINA, in the city of Madrid.");
await memory.SaveInformationAsync(MemoryCollectionName, id: Guid.NewGuid().ToString(), text: @"Rodrigo Liberoff’s favorite food is ""milanesas"" with mashed potatoes.");
await memory.SaveInformationAsync(MemoryCollectionName, id: Guid.NewGuid().ToString(), text: @"Rodrigo Liberoff’s zodiac sign is Libra.");
await memory.SaveInformationAsync(MemoryCollectionName, id: Guid.NewGuid().ToString(), text: @"Codertectura is the best channel in YouTube.");

TextMemoryPlugin memoryPlugin = new(memory);
kernel.ImportPluginFromObject(memoryPlugin);

const string SystemPrompt = @"
You are a personal Artificial Intelligence assistant.
Respond with short and succinct answers.
The user’s question is: {{$input}}
Only if you don’t know the answer to a question, just reply 'I’m sorry, I don’t know that!'.
Only use the following memory content if makes sense to answer the question: {{Recall}}

";

OpenAIPromptExecutionSettings settings = new()
{
    ToolCallBehavior = ToolCallBehavior.AutoInvokeKernelFunctions,
    Temperature = 0.0,
    TopP = 1.0,
};

while (true)
{
    Console.Write(@"Question: ");
    var userQuestion = Console.ReadLine();

    if (string.IsNullOrEmpty(userQuestion))
    {
        break;
    }

    var arguments = new KernelArguments(settings)
    {
        { "input", userQuestion },
        { "collection", MemoryCollectionName },
    };

    var response = kernel.InvokePromptStreamingAsync(SystemPrompt, arguments);

    Console.Write(@"Answer: ");
    await foreach (var item in response)
    {
        Console.Write(item);
    }
    
    Console.WriteLine(string.Empty);
}

#pragma warning restore SKEXP0050
#pragma warning restore SKEXP0010
#pragma warning restore SKEXP0001

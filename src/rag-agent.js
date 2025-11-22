import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { AlibabaTongyiEmbeddings } from "@langchain/community/embeddings/alibaba_tongyi";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { StateGraph, START, END, MessagesAnnotation } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { z } from "zod";

// Initialize DeepSeek LLM
const llm = new ChatOpenAI({
  model: "deepseek-chat",
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: {
    baseURL: "https://api.deepseek.com/v1",
  },
  temperature: 0,
});

// Initialize embeddings
const embeddings = new AlibabaTongyiEmbeddings({
  apiKey: process.env.ALIBABA_API_KEY,
});

// Connect to existing ChromaDB collection
const vectorStore = await Chroma.fromExistingCollection(embeddings, {
  collectionName: "nike_10k_2023",
  url: "http://localhost:8000",
});

// Define retrieval tool
const retrievalTool = new DynamicStructuredTool({
  name: "retrieve_documents",
  description:
    "Search and retrieve relevant documents from Nike's 2023 annual report. Use this tool when you need factual information about Nike's business, financials, or operations.",
  schema: z.object({
    query: z.string().describe("The search query to find relevant documents"),
    numResults: z
      .number()
      .optional()
      .default(4)
      .describe("Number of documents to retrieve (default: 4)"),
  }),
  func: async ({ query, numResults = 4 }) => {
    console.log(`\n🔍 Retrieving documents for query: "${query}"\n`);

    const results = await vectorStore.similaritySearchWithScore(query, numResults);

    if (results.length === 0) {
      return "No relevant documents found.";
    }

    const formattedResults = results
      .map(([doc, score], index) => {
        return `Document ${index + 1} (Similarity: ${score.toFixed(4)}):
Page: ${doc.metadata.pageNumber}
Content: ${doc.pageContent}
---`;
      })
      .join("\n\n");

    console.log(`✅ Retrieved ${results.length} documents\n`);
    return formattedResults;
  },
});

// Bind tools to LLM
const tools = [retrievalTool];
const llmWithTools = llm.bindTools(tools);

// Define agent node
async function callModel(state) {
  const { messages } = state;
  console.log("\n🤖 Agent is thinking...\n");

  const response = await llmWithTools.invoke(messages);

  return { messages: [response] };
}

// Define conditional edge logic
function shouldContinue(state) {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1];

  // If there are tool calls, continue to tools node
  if (lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
    console.log("\n🔧 Calling tools...\n");
    return "tools";
  }

  // Otherwise, end
  console.log("\n✅ Agent finished\n");
  return END;
}

// Build the graph
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", new ToolNode(tools))
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldContinue, {
    tools: "tools",
    [END]: END,
  })
  .addEdge("tools", "agent");

// Compile the graph
const app = workflow.compile();

// Run the agent
async function runAgent(query) {
  console.log("=".repeat(80));
  console.log("🚀 RAG Agent Demo with LangGraph");
  console.log("=".repeat(80));
  console.log(`\n📝 User Query: ${query}\n`);
  console.log("=".repeat(80));

  const finalState = await app.invoke({
    messages: [{ role: "user", content: query }],
  });

  console.log("=".repeat(80));
  console.log("\n💬 Final Response:");
  console.log("=".repeat(80));

  const lastMessage = finalState.messages[finalState.messages.length - 1];
  console.log(lastMessage.content);

  console.log("\n" + "=".repeat(80));
  console.log("✨ RAG Agent Demo Completed");
  console.log("=".repeat(80));

  return lastMessage.content;
}

// Example usage
const question = "What was Nike's revenue in 2023?";
await runAgent(question);

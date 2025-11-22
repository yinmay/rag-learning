import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { AlibabaTongyiEmbeddings } from "@langchain/community/embeddings/alibaba_tongyi";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

const COLLECTION_NAME = "wangEditor-doc";
const CHROMA_URL = "http://localhost:8000";

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

// RAG Prompt Template
const ragPromptTemplate = ChatPromptTemplate.fromTemplate(`
You are a helpful assistant for answering questions about wangEditor documentation.
Use the following context from the documentation to answer the question.
If you cannot find the answer in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Please provide a clear and detailed answer in Chinese (中文):
`);

async function ragQuery(question) {
  console.log("=".repeat(80));
  console.log("🔍 WangEditor Documentation RAG System");
  console.log("=".repeat(80));
  console.log(`\n📝 User Question: ${question}\n`);
  console.log("=".repeat(80));

  // Step 1: Connect to vector store
  console.log("\n📚 Step 1: Connecting to vector database...");
  const vectorStore = await Chroma.fromExistingCollection(embeddings, {
    collectionName: COLLECTION_NAME,
    url: CHROMA_URL,
  });
  console.log("✅ Connected to ChromaDB");

  // Step 2: Retrieve relevant documents
  console.log("\n🔎 Step 2: Retrieving relevant documents...");
  const retrievedDocs = await vectorStore.similaritySearchWithScore(question, 4);

  console.log(`✅ Retrieved ${retrievedDocs.length} relevant documents:\n`);

  retrievedDocs.forEach(([doc, score], index) => {
    console.log(`Document ${index + 1}:`);
    console.log(`  Similarity Score: ${score.toFixed(4)}`);
    console.log(`  Chunk Index: ${doc.metadata.chunkIndex}/${doc.metadata.totalChunks}`);
    console.log(`  Content Preview: ${doc.pageContent.substring(0, 100)}...`);
    console.log();
  });

  // Step 3: Prepare context for LLM
  console.log("📝 Step 3: Preparing context for LLM...");
  const context = retrievedDocs
    .map(([doc, score], index) => {
      return `[Document ${index + 1}] (Relevance: ${score.toFixed(4)})\n${doc.pageContent}`;
    })
    .join("\n\n---\n\n");

  console.log(`✅ Context prepared (${context.length} characters)\n`);

  // Step 4: Generate answer using LLM
  console.log("🤖 Step 4: Generating answer with DeepSeek...");

  const chain = ragPromptTemplate.pipe(llm).pipe(new StringOutputParser());

  const answer = await chain.invoke({
    context: context,
    question: question,
  });

  console.log("✅ Answer generated\n");
  console.log("=".repeat(80));
  console.log("💬 Answer:");
  console.log("=".repeat(80));
  console.log(answer);
  console.log("\n" + "=".repeat(80));

  return {
    question,
    answer,
    retrievedDocs: retrievedDocs.length,
    sources: retrievedDocs.map(([doc, score]) => ({
      chunkIndex: doc.metadata.chunkIndex,
      score: score.toFixed(4),
    })),
  };
}

// Additional function to show sources
async function ragQueryWithSources(question) {
  const result = await ragQuery(question);

  console.log("\n📚 Sources:");
  console.log("-".repeat(80));
  result.sources.forEach((source, index) => {
    console.log(`Source ${index + 1}: Chunk ${source.chunkIndex} (Score: ${source.score})`);
  });
  console.log("=".repeat(80));

  return result;
}

// Main execution
async function main() {
  try {
    const question = "wangEditor 如何配置上传图片";
    await ragQueryWithSources(question);

    console.log("\n✨ RAG Query Completed Successfully!");
  } catch (error) {
    console.error("\n❌ Error:", error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

main();
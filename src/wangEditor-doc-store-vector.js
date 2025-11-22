import "dotenv/config";
import fs from "fs";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { AlibabaTongyiEmbeddings } from "@langchain/community/embeddings/alibaba_tongyi";
import { ChromaClient } from "chromadb";

const COLLECTION_NAME = "wangEditor-doc";
const CHROMA_URL = "http://localhost:8000";

async function storeWangEditorDoc() {
  console.log("=".repeat(80));
  console.log("📚 WangEditor Documentation Vector Storage");
  console.log("=".repeat(80));

  // Step 1: Clear existing collection
  console.log("\n🗑️  Step 1: Clearing existing collection...");
  try {
    const chromaClient = new ChromaClient({ path: CHROMA_URL });

    // Try to delete the collection if it exists
    try {
      await chromaClient.deleteCollection({ name: COLLECTION_NAME });
      console.log(`✅ Deleted existing collection: ${COLLECTION_NAME}`);
    } catch (error) {
      console.log(`ℹ️  Collection ${COLLECTION_NAME} does not exist yet, creating new one`);
    }
  } catch (error) {
    console.log(`⚠️  Warning: ${error.message}`);
  }

  // Step 2: Load markdown file
  console.log("\n📖 Step 2: Loading markdown file...");
  const fileContent = fs.readFileSync("files/wangEditor-doc.md", "utf-8");
  const docs = [
    new Document({
      pageContent: fileContent,
      metadata: { source: "files/wangEditor-doc.md" },
    }),
  ];
  console.log(`✅ Loaded document with ${docs[0].pageContent.length} characters`);

  // Step 3: Split documents into chunks
  console.log("\n✂️  Step 3: Splitting document into chunks...");
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const splits = await textSplitter.splitDocuments(docs);
  console.log(`✅ Split into ${splits.length} text chunks`);

  // Clean metadata for ChromaDB compatibility
  const cleanedSplits = splits.map((doc, index) => ({
    pageContent: doc.pageContent,
    metadata: {
      source: doc.metadata.source || "files/wangEditor-doc.md",
      chunkIndex: index,
      totalChunks: splits.length,
    },
  }));

  // Step 4: Initialize embeddings
  console.log("\n🧮 Step 4: Initializing Alibaba Tongyi Embeddings...");
  const embeddings = new AlibabaTongyiEmbeddings({
    apiKey: process.env.ALIBABA_API_KEY,
  });

  // Step 5: Store to ChromaDB
  console.log("\n💾 Step 5: Storing vectors to ChromaDB...");
  const vectorStore = await Chroma.fromDocuments(cleanedSplits, embeddings, {
    collectionName: COLLECTION_NAME,
    url: CHROMA_URL,
  });
  console.log(`✅ Successfully stored ${cleanedSplits.length} chunks to collection: ${COLLECTION_NAME}`);

  console.log("\n" + "=".repeat(80));
  console.log("✨ Vector Storage Completed");
  console.log("=".repeat(80));

  return vectorStore;
}

async function runTests(vectorStore) {
  console.log("\n" + "=".repeat(80));
  console.log("🧪 Running Tests");
  console.log("=".repeat(80));

  // Test 1: Basic retrieval test
  console.log("\n📝 Test 1: Basic Retrieval Test");
  console.log("-".repeat(80));
  const testQuery1 = "What is wangEditor?";
  console.log(`Query: "${testQuery1}"`);

  const results1 = await vectorStore.similaritySearchWithScore(testQuery1, 3);
  console.log(`\n✅ Retrieved ${results1.length} results:`);

  results1.forEach(([doc, score], index) => {
    console.log(`\nResult ${index + 1}:`);
    console.log(`  Similarity Score: ${score.toFixed(4)}`);
    console.log(`  Chunk Index: ${doc.metadata.chunkIndex}/${doc.metadata.totalChunks}`);
    console.log(`  Content Preview: ${doc.pageContent.substring(0, 150)}...`);
  });

  // Test 2: Feature-specific query
  console.log("\n" + "-".repeat(80));
  console.log("\n📝 Test 2: Feature-Specific Query");
  console.log("-".repeat(80));
  const testQuery2 = "How to configure the editor?";
  console.log(`Query: "${testQuery2}"`);

  const results2 = await vectorStore.similaritySearchWithScore(testQuery2, 3);
  console.log(`\n✅ Retrieved ${results2.length} results:`);

  results2.forEach(([doc, score], index) => {
    console.log(`\nResult ${index + 1}:`);
    console.log(`  Similarity Score: ${score.toFixed(4)}`);
    console.log(`  Chunk Index: ${doc.metadata.chunkIndex}/${doc.metadata.totalChunks}`);
    console.log(`  Content Preview: ${doc.pageContent.substring(0, 150)}...`);
  });

  // Test 3: Collection statistics
  console.log("\n" + "-".repeat(80));
  console.log("\n📝 Test 3: Collection Statistics");
  console.log("-".repeat(80));

  const chromaClient = new ChromaClient({ path: CHROMA_URL });
  const collection = await chromaClient.getCollection({ name: COLLECTION_NAME });
  const count = await collection.count();

  console.log(`Collection Name: ${COLLECTION_NAME}`);
  console.log(`Total Documents: ${count}`);
  console.log(`✅ Collection is accessible and contains data`);

  console.log("\n" + "=".repeat(80));
  console.log("✅ All Tests Passed!");
  console.log("=".repeat(80));
}

// Main execution
async function main() {
  try {
    const vectorStore = await storeWangEditorDoc();
    await runTests(vectorStore);
  } catch (error) {
    console.error("\n❌ Error:", error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

main();
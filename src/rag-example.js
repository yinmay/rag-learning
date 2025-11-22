import "dotenv/config";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { AlibabaTongyiEmbeddings } from "@langchain/community/embeddings/alibaba_tongyi";

async function ragExample() {
  console.log("=== RAG Demo ===\n");

  // 1. Load PDF file
  console.log("1. Loading PDF file...");
  const loader = new PDFLoader("files/nke-10k-2023.pdf");
  const docs = await loader.load();
  console.log(`   Loaded ${docs.length} document pages\n`);

  // 2. Split documents into chunks
  console.log("2. Splitting documents into chunks...");
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const splits = await textSplitter.splitDocuments(docs);
  console.log(`   Split into ${splits.length} text chunks\n`);

  // Clean metadata, keep only simple types
  const cleanedSplits = splits.map((doc) => {
    const pageNumber = doc.metadata.loc?.pageNumber || 0;
    return {
      pageContent: doc.pageContent,
      metadata: {
        source: doc.metadata.source,
        pageNumber: pageNumber,
      },
    };
  });

  // 3. Create Embedding model
  console.log("3. Initializing Alibaba Tongyi Embeddings...");
  const embeddings = new AlibabaTongyiEmbeddings({
    apiKey: process.env.ALIBABA_API_KEY,
  });

  // 4. Store to ChromaDB
  console.log("4. Storing documents to ChromaDB...");
  const vectorStore = await Chroma.fromDocuments(cleanedSplits, embeddings, {
    collectionName: "nike_10k_2023",
    url: "http://localhost:8000",
  });
  console.log("   Storage completed\n");

  // 5. Execute two retrieval demos
  console.log("=== Retrieval Demo ===\n");

  // First retrieval
  console.log("Retrieval 1: Nike's Revenue");
  console.log("-".repeat(80));
  const results1 = await vectorStore.similaritySearchWithScore(
    "What is Nike's total revenue?",
    4
  );
  results1.forEach(([doc, score], index) => {
    console.log(`\nResult ${index + 1}:`);
    console.log(`Similarity Score: ${score.toFixed(4)}`);
    console.log(`Content: ${doc.pageContent.substring(0, 300)}...`);
    console.log(`Source: Page ${doc.metadata.pageNumber || "Unknown"}`);
    console.log(`Document:`, JSON.stringify(doc.metadata, null, 2));
  });

  console.log("\n" + "=".repeat(80) + "\n");

  // Second retrieval
  console.log("Retrieval 2: Nike's Market Strategy");
  console.log("-".repeat(80));
  const results2 = await vectorStore.similaritySearchWithScore(
    "Nike's digital strategy and innovation",
    4
  );
  results2.forEach(([doc, score], index) => {
    console.log(`\nResult ${index + 1}:`);
    console.log(`Similarity Score: ${score.toFixed(4)}`);
    console.log(`Content: ${doc.pageContent.substring(0, 300)}...`);
    console.log(`Source: Page ${doc.metadata.pageNumber || "Unknown"}`);
    console.log(`Document:`, JSON.stringify(doc.metadata, null, 2));
  });

  console.log("\n=== RAG Demo Completed ===");
}

// Run example
ragExample().catch(console.error);

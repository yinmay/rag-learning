import { ChromaClient } from 'chromadb'
import { AlibabaTongyiEmbeddings } from '@langchain/community/embeddings/alibaba_tongyi'
import 'dotenv/config'

const embedder = new AlibabaTongyiEmbeddings({})

// Custom embedding function adapter for Chroma
// Note: Chroma expects a generate method that returns embeddings for the given texts
const embeddingFunction = {
  generate: async (texts) => {
    return await embedder.embedDocuments(texts)
  },
}

const client = new ChromaClient()

const collection = await client.createCollection({
  name: 'my_collection_v2',
  embeddingFunction,
})

await collection.add({
  ids: ['id1', 'id2'],
  documents: [
    'This is a document about pineapple',
    'This is a document about oranges',
  ],
})

const results = await collection.query({
  queryTexts: ['This is a query document about hawaii'],
  nResults: 2,
})
console.log('Query results:', results)

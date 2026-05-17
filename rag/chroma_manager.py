import chromadb
from config import CHROMA_DB_DIR, EMBEDDING_DIM
from rag.embedding import MultiModalEmbedding


class ChromaManager:
    def __init__(self, collection_name: str = "chuhui_rag"):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: list, metadatas: list = None, ids: list = None):
        """添加文档到向量数据库"""
        embeddings = [MultiModalEmbedding.get_text_embedding(doc) for doc in documents]
        
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"已添加 {len(documents)} 条文档到Chroma数据库")

    def add_image(self, image_path: str, metadata: dict = None):
        """添加图片到向量数据库"""
        embedding = MultiModalEmbedding.get_image_embedding(image_path)
        content = f"[图片] {image_path}"
        
        if metadata is None:
            metadata = {}
        metadata['type'] = 'image'
        metadata['path'] = image_path
        
        doc_id = f"image_{hash(image_path)}"
        self.collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )
        print(f"已添加图片: {image_path}")

    def add_video(self, video_url: str, description: str, metadata: dict = None):
        """添加视频到向量数据库"""
        embedding = MultiModalEmbedding.get_video_embedding(video_url)
        content = f"[视频] {description}"
        
        if metadata is None:
            metadata = {}
        metadata['type'] = 'video'
        metadata['url'] = video_url
        metadata['description'] = description
        
        doc_id = f"video_{hash(video_url)}"
        self.collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )
        print(f"已添加视频: {description}")

    def query(self, query_text: str, n_results: int = 5, where: dict = None) -> dict:
        """查询向量数据库"""
        query_embedding = MultiModalEmbedding.get_text_embedding(query_text)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        return results

    def get_collection_stats(self) -> dict:
        """获取集合统计信息"""
        return self.collection.count()

    def delete_by_ids(self, ids: list):
        """根据ID删除文档"""
        self.collection.delete(ids=ids)

    def get_all_documents(self) -> list:
        """获取所有文档"""
        results = self.collection.get()
        docs = []
        if results and 'documents' in results and results['documents']:
            for i in range(len(results['documents'])):
                doc = {
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i] if results.get('metadatas') else {},
                    'id': results['ids'][i] if results.get('ids') else f"doc_{i}"
                }
                docs.append(doc)
        return docs

    def clear_collection(self):
        """清空集合"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
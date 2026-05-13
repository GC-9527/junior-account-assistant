import os
import base64
import numpy as np
import dashscope
from http import HTTPStatus
from config import DASHSCOPE_API_KEY, MULTIMODAL_EMBEDDING_MODEL, TEXT_EMBEDDING_MODEL

dashscope.api_key = DASHSCOPE_API_KEY


class MultiModalEmbedding:
    @staticmethod
    def get_text_embedding(text: str) -> list:
        """文本embedding - 使用text-embedding-v4模型"""
        resp = dashscope.TextEmbedding.call(
            model=TEXT_EMBEDDING_MODEL,
            input=text
        )
        if resp.status_code != HTTPStatus.OK:
            raise Exception(f"文本Embedding失败: {resp.message}")
        return resp.output['embeddings'][0]['embedding']

    @staticmethod
    def get_image_embedding(image_path: str) -> list:
        """图片embedding - 使用multimodal-embedding-v1模型"""
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')

        ext = os.path.splitext(image_path)[1].lower().lstrip('.')
        if ext == 'jpg':
            ext = 'jpeg'
        image_data = f"data:image/{ext};base64,{base64_image}"

        resp = dashscope.MultiModalEmbedding.call(
            model=MULTIMODAL_EMBEDDING_MODEL,
            input=[{'image': image_data}]
        )
        if resp.status_code != HTTPStatus.OK:
            raise Exception(f"图片Embedding失败: {resp.message}")
        return resp.output['embeddings'][0]['embedding']

    @staticmethod
    def get_video_embedding(video_url: str) -> list:
        """视频embedding（多帧取平均）"""
        resp = dashscope.MultiModalEmbedding.call(
            model=MULTIMODAL_EMBEDDING_MODEL,
            input=[{'video': video_url}]
        )
        if resp.status_code != HTTPStatus.OK:
            raise Exception(f"视频Embedding失败: {resp.message}")

        embeddings = resp.output['embeddings']
        if len(embeddings) > 1:
            vectors = [np.array(e['embedding']) for e in embeddings]
            return np.mean(vectors, axis=0).tolist()
        return embeddings[0]['embedding']

    @staticmethod
    def get_embedding(content: str, content_type: str = "text") -> list:
        """统一的embedding接口"""
        if content_type == "text":
            return MultiModalEmbedding.get_text_embedding(content)
        elif content_type == "image":
            return MultiModalEmbedding.get_image_embedding(content)
        elif content_type == "video":
            return MultiModalEmbedding.get_video_embedding(content)
        else:
            raise ValueError(f"不支持的内容类型: {content_type}")
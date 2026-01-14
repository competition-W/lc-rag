# 文件: /mnt/omicshub/rag/services/minio_client.py

from minio import Minio
from minio.error import S3Error
import io
import logging
from datetime import timedelta

from config import settings

logger = logging.getLogger(__name__)

class MinioClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MinioClient, cls).__new__(cls)
            cls._instance._init_client()
        return cls._instance

    def _init_client(self):
        # 处理 Endpoint，去除 http/https 前缀
        endpoint = settings.MINIO_ENDPOINT.replace("http://", "").replace("https://", "")
        
        self.client = Minio(
            endpoint=endpoint,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE
        )
        self.default_bucket = settings.MINIO_BUCKET_DOCUMENTS
        self._ensure_buckets()

    def _ensure_buckets(self):
        """确保配置的 Bucket 存在"""
        buckets = [settings.MINIO_BUCKET_DOCUMENTS, settings.MINIO_BUCKET_PROCESSED]
        for bucket in buckets:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    logger.info(f"✅ Created MinIO bucket: {bucket}")
            except S3Error as e:
                logger.error(f"❌ MinIO bucket check failed for {bucket}: {e}")
                pass

    def upload_file(self, object_name: str, file_data: bytes, content_type: str = "application/octet-stream") -> str:
        """上传文件到文档 Bucket"""
        try:
            data_stream = io.BytesIO(file_data)
            self.client.put_object(
                bucket_name=self.default_bucket,
                object_name=object_name,
                data=data_stream,
                length=len(file_data),
                content_type=content_type
            )
            logger.info(f"⬆️ Uploaded to MinIO: {self.default_bucket}/{object_name}")
            return f"{self.default_bucket}/{object_name}"
        except Exception as e:
            logger.error(f"Failed to upload file {object_name}: {e}")
            raise

    def download_file(self, object_name: str, bucket_name: str = None) -> bytes:
        """下载文件"""
        bucket = bucket_name or self.default_bucket
        try:
            response = self.client.get_object(bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except Exception as e:
            logger.error(f"Failed to download file {bucket}/{object_name}: {e}")
            raise

    def get_presigned_url(self, object_name: str, bucket_name: str = None, expires_hours: int = 1) -> str:
        """
        生成预签名 URL（临时访问链接）
        
        Args:
            object_name: 对象路径
            bucket_name: Bucket 名称（可选，默认使用 default_bucket）
            expires_hours: 过期时间（小时）
            
        Returns:
            presigned_url: 预签名 URL
        """
        bucket = bucket_name or self.default_bucket
        
        url = self.client.presigned_get_object(
            bucket,  
            object_name,
            expires=timedelta(hours=expires_hours)
        )
        
        logger.info(f"生成预签名 URL: {bucket}/{object_name} (有效期: {expires_hours}h)")
        return url


# 全局单例
minio_client = MinioClient()

from rag_assistant.core.config import Settings
from rag_assistant.indexing.storage import LocalObjectStore, ObjectStore, S3ObjectStore


def create_object_store(settings: Settings) -> ObjectStore:
    """Build the configured source store and reject local storage in production."""
    if settings.object_store_backend == "local":
        if settings.app_env == "production":
            raise ValueError("Production requires OBJECT_STORE_BACKEND=s3")
        return LocalObjectStore(settings.local_object_store_root)
    if settings.s3_bucket is None:
        raise ValueError("S3_BUCKET is required when OBJECT_STORE_BACKEND=s3")
    return S3ObjectStore(
        bucket=settings.s3_bucket,
        endpoint_url=str(settings.s3_endpoint_url) if settings.s3_endpoint_url else None,
        region=settings.s3_region,
        access_key_id=(
            settings.s3_access_key_id.get_secret_value() if settings.s3_access_key_id else None
        ),
        secret_access_key=(
            settings.s3_secret_access_key.get_secret_value()
            if settings.s3_secret_access_key
            else None
        ),
    )

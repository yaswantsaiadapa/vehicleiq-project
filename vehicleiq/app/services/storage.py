"""
VehicleIQ — CloudStorageService
================================
Manages AWS S3-compatible cloud object storage (Supabase Storage / Cloudflare R2 / AWS S3)
using the official boto3 SDK.

Features:
  - Streams vehicle crop images and videos directly to the cloud bucket
  - Generates public CDN / presigned HTTPS URLs for frontend rendering
  - Automatic graceful fallback to local storage if offline or keys missing
"""

import logging
import os
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

from app import config

log = logging.getLogger(__name__)


class CloudStorageService:
    """
    AWS S3-compatible cloud object storage wrapper.
    Seamlessly works with Supabase Storage, Cloudflare R2, and Amazon S3.
    """

    def __init__(self) -> None:
        self.endpoint_url = os.environ.get("S3_ENDPOINT_URL") or os.environ.get("SUPABASE_S3_ENDPOINT")
        self.access_key = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("SUPABASE_S3_ACCESS_KEY_ID")
        self.secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("SUPABASE_S3_SECRET_ACCESS_KEY")
        self.bucket_name = os.environ.get("S3_BUCKET_NAME", "vehicleiq-crops")
        self.region_name = os.environ.get("S3_REGION_NAME", "us-east-1")
        self.public_domain = os.environ.get("S3_PUBLIC_DOMAIN", "").rstrip("/")

        self._s3_client = None
        self.is_cloud_enabled = False

        if self.endpoint_url and self.access_key and self.secret_key:
            try:
                self._s3_client = boto3.client(
                    "s3",
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=self.access_key.strip(),
                    aws_secret_access_key=self.secret_key.strip(),
                    region_name=self.region_name,
                    config=Config(signature_version="s3v4"),
                )
                self.is_cloud_enabled = True
                log.info(
                    "CloudStorageService initialized: Connected to bucket '%s' at %s",
                    self.bucket_name,
                    self.endpoint_url,
                )
            except Exception as e:
                log.warning("Failed to initialize cloud storage client: %s. Using local disk fallback.", e)
                self.is_cloud_enabled = False
        else:
            log.info("Cloud storage credentials not configured. Using local filesystem storage.")

    def upload_crop(self, local_file_path: Path, vehicle_id: str) -> str | None:
        """
        Uploads a vehicle crop to cloud bucket.
        Returns the public HTTPS URL, or None if fallback.
        """
        if not self.is_cloud_enabled or not local_file_path.exists():
            return None

        file_name = local_file_path.name
        key = f"crops/{vehicle_id}/{file_name}"

        try:
            self._s3_client.upload_file(
                Filename=str(local_file_path),
                Bucket=self.bucket_name,
                Key=key,
                ExtraArgs={"ContentType": "image/jpeg"},
            )
            return self.get_crop_url(vehicle_id, file_name)
        except Exception as e:
            log.warning("Cloud upload failed for %s: %s", local_file_path, e)
            return None

    def get_crop_url(self, vehicle_id: str, file_name: str) -> str:
        """Build public HTTPS CDN URL for a vehicle crop."""
        if self.public_domain:
            return f"{self.public_domain}/crops/{vehicle_id}/{file_name}"

        # If Supabase storage, build standard public URL
        if self.endpoint_url and "supabase.co" in self.endpoint_url:
            # Extract project ref from endpoint e.g. https://imoizjabszamxkavarlg.storage.supabase.co
            project_ref = self.endpoint_url.split("//")[1].split(".")[0]
            return f"https://{project_ref}.supabase.co/storage/v1/object/public/{self.bucket_name}/crops/{vehicle_id}/{file_name}"

        # Fallback to standard S3 public path
        return f"{self.endpoint_url}/{self.bucket_name}/crops/{vehicle_id}/{file_name}"

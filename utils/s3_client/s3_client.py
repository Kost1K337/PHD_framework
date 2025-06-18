import zipfile
import boto3
import json
import io

from botocore.exceptions import ClientError

from src.s3_client.s3_exceptions import UploadFileError, BucketCreationError


class S3Client:
    def __init__(self,
                 storage_url: str,
                 result_bucket: str,
                 access_key: str,
                 secret_key: str):
        self._result_bucket = result_bucket
        self._storage_url = storage_url
        self._access_key = access_key
        self._secret_key = secret_key

        self._s3 = boto3.client(
            "s3",
            endpoint_url=self._storage_url,
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            verify=False
        )

        try:
            self._s3.head_bucket(Bucket=result_bucket)
        except ClientError:
            try:
                self._s3.create_bucket(Bucket=result_bucket)
            except BucketCreationError(f"Не удалось создать бакет {result_bucket}") as e:
                raise e

    def push_to_result_bucket(self, result_data, file_name: str):
        json_data = json.dumps(result_data).encode("utf-8")
        # Запись архива
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(f"{file_name}.json", json_data)
        zip_buffer.seek(0)
        zip_file_name = f"{file_name}.zip"
        # Запись в S3
        try:
            self._s3.put_object(Body=zip_buffer.getvalue(),
                                Bucket=self._result_bucket,
                                Key=zip_file_name,
                                ContentType="application/zip")
        except UploadFileError(f"Не удалось загрузить файл {zip_file_name}") as e:
            raise e

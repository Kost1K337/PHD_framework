class UploadFileError(BaseException):
    def __init__(self, msg="Не удалось загрузить файл в хранилище S3"):
        super().__init__(msg)


class BucketCreationError(BaseException):
    def __init__(self, msg):
        super().__init__(msg)

import io

from google.cloud import storage
from google.cloud.storage import Blob
from google.oauth2 import service_account
from settings import GOOGLE_SERVICE_ACCOUNT_CREDENTIAL, STORAGE_BUCKET

AUTO_SEGMENT_TUNER_HYPER_PARAMETER_GCS_PATH = (
    "{STORAGE_BUCKET}/model_tuner/hyper_parameter"
)
AUTO_SEGMENT_TRAIN_MODEL_GCS_URI = (
    "gs://{STORAGE_BUCKET}/model_trained/auto_segment_model/"
)
AUTO_SEGMENT_TRAIN_MODEL_GCS_PATH = "model_trained/models/model.h5"
AUTO_SEGMENT_LOG_PATH = "logs/{task}/{datetime}/log.csv"


credentials = service_account.Credentials.from_service_account_info(
    GOOGLE_SERVICE_ACCOUNT_CREDENTIAL
)

storage_client = storage.Client(
    credentials=credentials,
    project=credentials.project_id,
)


def upload_blob(
    client: storage.Client,
    upload_path: str,
    upload_file: str,
    make_public: bool = False,
    cache_control: str = None,
) -> str:
    bucket = client.get_bucket(STORAGE_BUCKET)
    blob = bucket.blob(upload_path)
    blob.upload_from_file(upload_file)
    if make_public:
        blob.make_public()
    if cache_control:
        blob.cache_control = cache_control
        blob.patch()
    return blob.name


def download_blob(client: storage.Client, buffer: io.BytesIO, blob_path: str) -> bool:
    bucket = client.get_bucket(STORAGE_BUCKET)
    blob = bucket.get_blob(blob_path)
    if not blob:
        return False
    client.download_blob_to_file(blob, buffer)
    return True


def blob_exists(client: storage.Client, uri: str) -> bool:
    return Blob.from_string(uri, client).exists()

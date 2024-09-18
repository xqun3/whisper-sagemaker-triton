from pydantic import BaseModel

import boto3


account_id =boto3.client('sts').get_caller_identity().get('Account')
my_session = boto3.session.Session()
region = my_session.region_name
# print("account_id: ", account_id)
WORKING_DIR="/tmp"

def get_bucket_and_key(s3uri):
    """
    get_bucket_and_key is helper function
    """
    pos = s3uri.find('/', 5)
    bucket = s3uri[5: pos]
    key = s3uri[pos + 1:]
    return bucket, key


class InferenceOpt(BaseModel):
    language: str="",
    repo_id: str="whisper-large-v3",
    decoding_method: str = "greedy_search",
    whisper_prompt: str="",
    s3_path: str=""

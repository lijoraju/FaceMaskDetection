from huggingface_hub import HfApi
import os

api = HfApi()

repo_id = "lijoraju/face_mask_detector_model" 
local_file_path = os.path.join(".", "models", "final", "face_mask_detector_version01.pth")

api.upload_file(
    path_or_fileobj=local_file_path,
    path_in_repo="face_mask_detector_version01.pth",
    repo_id=repo_id,
    repo_type="model",
)

print(f"Model uploaded to {repo_id}")
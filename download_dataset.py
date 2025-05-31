import os
from huggingface_hub import snapshot_download

# Replace with your dataset repo ID
repo_id = "Jerry999/multilingual-terminology"

# Choose your target folder (absolute or relative)
local_dir = "./data"
os.makedirs(local_dir, exist_ok=True)

# Download entire repo as-is into flat directory
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False  # ensures actual files are copied, not symlinked
)

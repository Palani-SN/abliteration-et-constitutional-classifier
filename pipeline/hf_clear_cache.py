from huggingface_hub import scan_cache_dir

# Scan your Hugging Face cache directory
cache_info = scan_cache_dir()

# Create a strategy to delete everything found
delete_strategy = cache_info.delete_revisions(*[
    revision.commit_hash 
    for repo in cache_info.repos 
    for revision in repo.revisions
])

# Execute the deletion and free disk space
delete_strategy.execute()
print("All local Hugging Face models removed.")

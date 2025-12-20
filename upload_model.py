from huggingface_hub import HfApi

api = HfApi()

api.create_repo(
    repo_id="anthonydavidson/balihotel-sentiment-deberta",
    repo_type="model",
    private=False  # keep it public for free
)

api.upload_folder(
    folder_path="best_sentiment_model",
    repo_id="anthonydavidson/balihotel-sentiment-deberta",
    repo_type="model"
)

api.create_repo(
    repo_id="anthonydavidson/balihotel-bart-summarizer",
    repo_type="model",
    private=False
)

api.upload_folder(
    folder_path="bart_model",
    repo_id="anthonydavidson/balihotel-bart-summarizer",
    repo_type="model"
)

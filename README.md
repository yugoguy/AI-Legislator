Setup
```
conda create -n ai-giin python=3.11 -y
conda activate ai-giin

conda install -c conda-forge \
  requests pandas python-dotenv beautifulsoup4 pillow pymupdf markdown weasyprint black -y

pip install openai anthropic backoff shutup
```

```
conda activate ai-giin
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export ESTAT_APP_ID="your_estat_app_id"
```

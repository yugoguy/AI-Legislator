Environment Setup
```
conda create -n ai-giin python=3.11 -y
conda activate ai-giin

conda install -c conda-forge \
  requests pandas python-dotenv beautifulsoup4 pillow pymupdf markdown weasyprint black -y

pip install openai anthropic backoff shutup
```

Environment Activation
```
conda activate ai-giin
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export ESTAT_APP_ID="your_estat_app_id"
```

Alternatively, create encrypted API key file for faster API setup
```
cat > keys.txt
# type in:
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export ESTAT_APP_ID="..."

gpg -c keys.txt
rm keys.txt
```
```
conda activate ai-giin
source <(gpg -dq keys.txt.gpg)
```

End
```
conda deactivate
gpgconf --kill gpg-agent
```

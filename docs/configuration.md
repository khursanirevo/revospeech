# Configuration

## API keys

For cloud API backends (revolab), set your API key:

```bash
# Option 1: Environment variable
export REVOLAB_API_KEY=rv-your-key

# Option 2: CLI command (saves to ~/.config/revospeech/config.yaml)
revospeech config set-api-key
```

Resolution order: constructor arg > env var > config file.

## Catalog source

```bash
export REVOS_CATALOG_REPO="myorg/revospeech"
```

Or in `~/.config/revospeech/config.yaml`:

```yaml
catalog_repo: "myorg/revospeech"
```

## Cache location

Models are cached in `~/.cache/revospeech/`. Override with:

```bash
export REVOS_CACHE_DIR=/path/to/cache
```

## Logging

```bash
revospeech --verbose <cmd>   # DEBUG
revospeech --quiet <cmd>     # WARNING
```

Or programmatically:

```python
import logging
logging.getLogger("revospeech").setLevel(logging.DEBUG)
```

## Auto-download

By default, models download automatically on first use:

```python
# Default behavior
asr = ASR("model-name")  # downloads if needed

# Disable for air-gapped environments
asr = ASR("model-name", auto_download=False)
```

## Color output

CLI uses color when stdout is a TTY. Disable with:

```bash
export NO_COLOR=1
revospeech models  # plain text
```

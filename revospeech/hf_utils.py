"""Shared HuggingFace download utilities with auth error handling."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def download_gated_model(
    repo_id: str,
    local_dir: str | Path,
) -> Path:
    """Download a (potentially gated) HuggingFace model with clear auth error messages.

    Raises RuntimeError with actionable guidance on 401/403 errors.
    """
    from huggingface_hub import snapshot_download

    local_dir = Path(local_dir)
    logger.info("Downloading model from %s", repo_id)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
        )
    except Exception as e:
        raise wrap_hf_error(e, repo_id) from e
    return local_dir


def get_hf_user() -> str | None:
    """Return the authenticated HuggingFace username, or None."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().whoami()
        return info.get("name") if isinstance(info, dict) else None
    except Exception:
        return None


def wrap_hf_error(error: Exception, repo_id: str) -> Exception:
    """Convert raw HF errors into user-friendly RuntimeErrors."""
    err = str(error).lower()
    if "403" in err or "access" in err or "permission" in err:
        return RuntimeError(
            f"Access denied to model '{repo_id}'.\n"
            f"Your HuggingFace account does not have access to "
            f"this gated model.\n"
            f"Request access at: "
            f"https://huggingface.co/{repo_id}\n"
            f"Then wait for the repository owner to approve."
        )
    if (
        "401" in err or "authentication" in err
        or "credentials" in err or "gated" in err
    ):
        return RuntimeError(
            f"Cannot access model '{repo_id}' — "
            f"it requires HuggingFace authentication.\n"
            f"Log in with:  huggingface-cli login\n"
            f"Or set:        export HF_TOKEN=your_token\n"
            f"Get a token:  https://huggingface.co/settings/tokens"
        )
    return error

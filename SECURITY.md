# Security Policy

## Reporting a Vulnerability

The RevoSpeech team takes security bugs seriously. We appreciate your efforts to
responsibly disclose your findings, and will make every effort to acknowledge
your contributions.

### How to Report

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please report suspected vulnerabilities privately to:
**sani@khursani.dev**

To help us triage and fix the issue quickly, please include:

- A clear description of the vulnerability and its potential impact
- The exact version(s) of `revospeech` affected (output of `revospeech info`)
- Step-by-step instructions to reproduce the issue
- A minimal proof-of-concept (script, command, or input file) if possible
- Any relevant logs, stack traces, or screenshots
- Your assessment of severity (low / medium / high / critical)
- Whether you have a proposed fix or patch

Please encrypt sensitive reports using our PGP key if available, and avoid
sharing exploit details in public channels (GitHub issues, discussions, Discord,
social media) until a fix has been released.

## Supported Versions

We currently provide security updates for the following versions of RevoSpeech:

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |
| < 1.0   | :x:                |

Pre-release versions (alphas, betas, release candidates) are not eligible for
security fixes. Please upgrade to a stable release before reporting a
vulnerability.

## Response Time SLA

We aim to respond to security reports on the following schedule:

| Severity | Acknowledgment | Initial Assessment | Fix / Patch Release     |
| -------- | -------------- | ------------------ | ----------------------- |
| Critical | Within 48 hours | Within 5 business days | Within 30 days        |
| High     | Within 48 hours | Within 5 business days | Within 60 days        |
| Medium   | Within 72 hours | Within 10 business days | Within 90 days       |
| Low      | Within 5 business days | Best effort    | Best effort             |

Times are estimates from the date we confirm the report. Complex fixes may take
longer; we will keep you informed of progress at least once a week until the fix
is released.

## Scope

### In Scope

- The `revospeech` Python package (source code in this repository)
- The `revospeech` package distributed via [PyPI](https://pypi.org/project/revospeech/)
- GitHub Actions workflows defined in `.github/workflows/`
- The `revospeech` CLI and its subcommands
- Code that processes user-supplied audio files or model configurations
- Dependency resolution and model download paths

### Out of Scope

- Third-party model weights and checkpoints (e.g., those hosted on Hugging Face
  Hub by upstream projects such as Sherpa-ONNX, Piper, Whisper, etc.)
- Issues in upstream dependencies (please report to the respective upstream
  project)
- User-modified installations or forks
- Self-hosted deployments with custom configurations that deviate from the
  documented setup
- Vulnerabilities requiring an attacker to already have root/admin access to the
  target machine
- Denial-of-service attacks against the project's public infrastructure (CI,
  PyPI mirror, etc.) — report these to the respective hosting provider
- Social engineering attacks against maintainers or contributors

If you are unsure whether your finding is in scope, please report it privately
and we will triage accordingly.

## Reward / Recognition

RevoSpeech is an open-source project and does not offer a monetary bug bounty.
However, we recognize and thank security researchers in the following ways:

- **Acknowledgment in release notes** — Contributors who report valid security
  issues will be credited (with permission) in the `CHANGELOG.md` entry for the
  fixing release.
- **Co-authorship on the fix** — If you submit a patch, you will be listed as a
  co-author on the relevant commit.
- **Hall of Fame** — With your consent, your name/handle and a brief description
  of the finding will be added to a security acknowledgments section.

If you wish to remain anonymous, we will respect that and credit you only as
"anonymous reporter" or omit credit entirely at your request.

## Disclosure Policy

- We follow a **coordinated disclosure** process.
- We will not publicly disclose the details of a vulnerability until a fix has
  been released and users have had reasonable time to upgrade.
- We may publish a security advisory on GitHub (using the
  [GitHub Security Advisories](https://github.com/khursanirevo/revospeech/security/advisories)
  feature) with a CVE assignment if the issue is significant.
- We will credit you in the advisory unless you request otherwise.

## Contact

For any questions about this security policy, contact:
**sani@khursani.dev**

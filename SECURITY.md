# Security Policy

## Supported Versions

The following versions of the Dia TTS model and its associated code are currently supported with security updates:

| Version | Supported          |
|---------|--------------------|
| main    | ✅                 |
| v1      | ✅                 |
| < v1    | ❌ (End of Support) |

## Reporting a Vulnerability

We take the security of Dia seriously, including vulnerabilities in the code, model weights, dependencies, or potential misuse of the model. If you discover a security issue, please report it privately to ensure responsible disclosure.

- **How to Report**:
  - Email: security@nari-labs.com (preferred for sensitive issues).
  - GitHub Issue: For non-sensitive issues, open a GitHub issue with the `[Security]` prefix in the title.
  - Discord: Contact a maintainer via the Nari Labs Discord server (link in README) for urgent issues.

- **What to Include**:
  - A detailed description of the vulnerability (e.g., code flaw, dependency issue, model misuse scenario).
  - Steps to reproduce the issue, if applicable.
  - Potential impact (e.g., data exposure, malicious audio generation).
  - Any suggested fixes or mitigations.

- **Response Process**:
  - We will acknowledge receipt of your report within 48 hours.
  - We aim to investigate and provide an initial response within 7 days.
  - If confirmed, we will work on a fix and coordinate a public disclosure timeline with you, typically within 30 days.
  - For critical vulnerabilities, we may issue an immediate patch and notify users via GitHub and Discord.

Please do not disclose vulnerabilities publicly until we have coordinated a fix and disclosure plan.

## Responsible Use

Dia is designed for research and educational purposes. The following uses are strictly prohibited, as outlined in the README:

- **Identity Misuse**: Generating audio resembling real individuals without explicit permission.
- **Deceptive Content**: Creating misleading or fake audio (e.g., for fraud or disinformation).
- **Illegal or Malicious Use**: Any use violating applicable laws or causing harm.

If you suspect misuse of Dia, report it to security@nari-labs.com with details of the incident.

## Dependency Security

Dia relies on external libraries (e.g., PyTorch, Gradio) and the Descript Audio Codec. To mitigate supply chain risks:

- We recommend using pinned dependency versions (see `requirements.txt` when available).
- Regularly check for updates to dependencies and apply security patches.
- Report any known vulnerabilities in dependencies to the maintainers.

## Code Contributions

All pull requests are reviewed by maintainers to ensure code integrity. Contributors are encouraged to:

- Test code thoroughly before submitting.
- Disclose any potential security implications in the pull request description.
- Avoid including sensitive data (e.g., API keys, audio samples) in commits.

## Model Weight Integrity

Pretrained model weights are hosted on Hugging Face. To ensure integrity:

- Verify checksums (SHA256) provided in the Hugging Face model page before use.
- Report any discrepancies or tampering concerns to security@nari-labs.com.

## Data Privacy

When using voice cloning or audio prompts:

- Avoid uploading audio containing sensitive personal information.
- Use the Gradio UI or Hugging Face Space responsibly, ensuring compliance with data protection laws (e.g., GDPR, CCPA).
- Report any data exposure risks in the UI or inference pipeline.

## Contact

For questions about this policy or ongoing security efforts, reach out to security@nari-labs.com or join our Discord server for community support.

Thank you for helping keep Dia safe and secure!

# Genie AI Document Compare Marketing Generator

Streamlit app for generating LinkedIn, newsletter, and blog content for the Genie AI Document Compare launch. Includes automatic source loading, quote extraction, per-channel drafts with feedback-and-regenerate loops, and final export.

## Quick start
1) Install dependencies:
```bash
pip install -r requirements.txt
```
2) Set OpenAI API key in the sidebar when running the app.
3) Run the app:
```bash
streamlit run app.py
```
4) Workflow:
   - Step 1: Review preloaded source materials (DOCX/PDF) and optionally upload your own files.
   - Step 2: Select which outputs to generate (LinkedIn, Newsletter, Blog); pick per-channel word counts.
   - Step 4: Extract quotes (sorted by impact) and select any you want.
   - Step 5: Review drafts per channel, add feedback in the chat, and regenerate. Approve when ready.
   - Step 6: Final content auto-generates for all approved (or latest) drafts; download per-piece.

## Models
- Primary: `gpt-4o-mini`
- Fallbacks (ordered): `gpt-5`, then `o4-mini-2025-04-16`
- Temperature is dropped for models that disallow it (e.g., gpt-5, o4-mini snapshots) to avoid 400 errors.

## Branding (built-in)
- Genie AI brand: trust, clarity, simplicity.
- Primary color: `#673ab7`; Secondary: `#007aff`.
- Tone/CTA guidance in the brand guidelines and prompts.

## Notes
- Feedback chat per channel is always visible in Step 5. Sending feedback triggers regeneration with a status indicator and app rerun.
- Finalization in Step 6 auto-runs for all approved drafts; if a channel has no approved draft, the latest version is finalized to ensure nothing is missing.
- Debug log (sidebar) shows newest entries first. Check it for API errors or fallback model usage.

## Optional: Basic chat demo
```bash
streamlit run basic_chat.py
```
Contains echo, simple streaming, ChatGPT-like, and file-aware chat tabs.

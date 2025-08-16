# Awesome MI theory (MkDocs)

Simple site with About + Contacts and a menu of articles. LaTeX via MathJax, images supported.

## Local preview

```bash
python -m venv .venv
# Linux/macOS
. .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
mkdocs serve
```

Open http://127.0.0.1:8000

## Deploy to GitHub Pages (recommended)

1. Push this project to the `main` branch of your GitHub repo.
2. In **Settings → Pages**, set **Source = GitHub Actions**.
3. Push any commit to `main` to trigger the workflow.

The site will be available at:
```
https://sadsabrina.github.io/awesome-mi-theory/
```

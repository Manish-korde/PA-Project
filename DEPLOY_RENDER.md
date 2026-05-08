# Deploy to Render (Free Tier)

This project runs a Python web service (`app.py`) that serves the UI from `web/` and exposes `/api/*` endpoints.

## Option A (Recommended): Blueprint with `render.yaml`

1. Push this repo to GitHub.
2. In Render, click **New** → **Blueprint**.
3. Select your repository and confirm.
4. Render will read `render.yaml` and create a **Free** web service.

## Option B: Manual setup (no blueprint)

1. In Render, click **New** → **Web Service**.
2. Connect the GitHub repo.
3. Configure:
   - **Runtime:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
   - **Plan:** Free
4. Environment variables:
   - `HOST=0.0.0.0`
   - `PORT=8000` (Render also provides `PORT` automatically; either is fine)

## Notes (Free tier)

- Free web services can **spin down** when idle; the next visit can be slow (cold start).
- Installing `torch` / `torchvision` can make deploys slow on free tiers. If builds fail due to size/time, consider:
  - Switching to a paid instance, or
  - Moving heavy ML dependencies behind a separate service, or
  - Using a lighter CPU-only dependency set for demo deployments.


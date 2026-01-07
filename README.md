# Malaysian Sign Language (MSL) AI Prototype

This project is a simple full-stack template for our **Computer Vision group assignment**.
It includes:

- `backend/` – Flask API with a dummy `/predict` endpoint
- `frontend/` – React (Vite) web app to upload an MSL video and show translation
- `static/` – placeholder for static assets (logos, sample videos, etc.)

## Quick start

### 1. Backend

```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python app.py

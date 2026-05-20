# Context for SoilIntel Image Enhancement Pipeline

## 1. Objective
We are integrating a high-fidelity **Real-ESRGAN (Super-Resolution GAN)** image enhancement pipeline into the SoilIntel backend. This replaces previous simulated CSS/PIL-based effects. The goal is to provide a production-ready generative AI solution to clarify blurry soil samples, enabling the agronomist UI to render high-resolution reconstructions.

## 2. File Structure & Where to Look (Token Optimization)
To save context tokens, **DO NOT** read the entire repository. The current feature integration is strictly localized to the following files:

- **`soil_enhancer.py`**
  - **What's inside:** Core `SoilImageEnhancer` wrapper class for Real-ESRGAN.
  - **Look here if:** Inference fails, PyTorch throws memory/CUDA errors, or you need to optimize the model inference parameters (e.g., `half=True`, `tile` settings).
- **`app.py`**
  - **What's inside:** The Python custom HTTP server.
  - **Look here if:** You need to debug the `/api/enhance-image` endpoint inside the `do_POST` method (around line 875), specifically concerning Base64 image payload parsing and formatting.
- **`web/app.js`**
  - **What's inside:** Frontend JavaScript logic.
  - **Look here if:** You need to debug the frontend `fetch` request triggered by the "Enhance Image" button (around line 785) or the DOM injection of the returned image.
- **`requirements.txt`**
  - **What's inside:** Python dependencies.
  - **Look here if:** You need to verify PyTorch, `realesrgan`, `basicsr`, or `gfpgan` versions.
- **`web/index.html`**
  - **What's inside:** DOM structure. (Unlikely to need edits).
  - **Look here if:** You need to verify element IDs (`#enhance-btn`, `#gan-augmentation-grid`).

## 3. Current Implementation Status
### Dependencies
- Added `realesrgan`, `basicsr`, and `gfpgan` to `requirements.txt`. The system relies on PyTorch for inference.

### Core Logic (`soil_enhancer.py`)
- Created a `SoilImageEnhancer` class that wraps the `RealESRGAN_x4plus` model.
- It includes lazy loading and automatic weight downloading (into `~/.cache/torch/` or similar) upon first use.
- The `enhance_image` method accepts a PIL Image and returns an upscaled, sharpened PIL Image.

### Backend API (`app.py`)
- Created a new POST endpoint `/api/enhance-image` under the `do_POST` method.
- The endpoint expects a JSON payload containing the `image` key with a base64-encoded string.
- It strips any `data:image/...` prefix, decodes the base64 string, processes it via `SoilImageEnhancer`, and returns a JSON response containing `{"enhanced_image": "data:image/jpeg;base64,..."}`.

### Frontend (`web/app.js` & `web/index.html`)
- Added an **"Enhance Image"** button to the main visual identification form in `index.html`.
- In `app.js`, an event listener reads the file input via `FileReader`, converts it to base64, and sends it to `/api/enhance-image` via a `fetch` `POST` request.
- The response is dynamically injected into the "Visual Decision Support" (GAN Trace) section (`#gan-augmentation-grid`).

## 4. Issues Encountered & Resolved So Far
1. **Accidental Import Deletion**: `SoilAnalysisService` was accidentally removed from `app.py` causing a `NameError` during startup. **(Fixed)**
2. **HTTP 414 (Request-URI Too Long)**: The frontend was initially sending the base64 image as a `GET` request query parameter (`/api/enhance-image?image=...`), causing URI limits to be exceeded and triggering a `SyntaxError`. **(Fixed)**
    - *Resolution*: Moved the endpoint to `do_POST` in `app.py` and updated `fetch` in `app.js` to send the payload in a JSON body.

## 5. Instructions for Codex / Where to Continue Debugging
We have just fixed the `414 Request-URI Too Long` error. The POST request *should* now successfully reach the server. However, you will need to monitor and debug the following potential failure points during live testing:

1. **PyTorch/ESRGAN Inference Issues (`soil_enhancer.py`)**: 
    - Watch for Out-Of-Memory (OOM) errors or CPU/GPU fallback issues during `enhancer.enhance_image(img)`. 
    - The `RealESRGANer` model can be highly resource-intensive. If inference crashes or hangs, you may need to implement tiling or downscaling before inference.
2. **Base64 Payload Decoding/Encoding Constraints (`app.py`)**: 
    - Ensure that the JSON base64 string doesn't contain unexpected artifacts and correctly strips the data URI prefix (e.g., `data:image/jpeg;base64,`).
    - Verify that Python's `base64.b64decode` handles the parsed JSON string correctly.
3. **Model Weight Downloading (`soil_enhancer.py`)**: 
    - On the very first run, it downloads the pre-trained `.pth` weights. Monitor for timeout issues or network failures during this phase.

**To Test:** Start the server (`python app.py`), use the "Visual Scan" tab, click "Enhance Image", and monitor the terminal logs and browser console.

import json
import os
import urllib.parse
import urllib.request
import urllib.error
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import joblib
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_ROOT / ".env", override=True)
load_dotenv(override=True)

def _read_api_key(name: str) -> str:
    value = os.getenv(name) or ""
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1].strip()
    return value

GROQ_API_KEY = _read_api_key("GROQ_API_KEY") or _read_api_key("GROK_API_KEY")

def _http_json(url: str, payload: dict[str, Any], headers: Optional[dict[str, str]] = None, timeout_s: int = 30) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if headers:
        for key, value in headers.items():
            req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {exc.code}: {body or exc.reason}") from None


class CropModelTool:
    def __init__(self) -> None:
        self._loaded = False
        self._metrics: dict[str, Any] = {}
        self._label_encoder = None
        self._rf_model = None
        self._scaler = None
        self._ann_state_path: Optional[Path] = None
        self._best_model: Optional[str] = None
        self._classes: list[str] = []

        self._torch = None
        self._CropANN = None

    def _load(self) -> None:
        if self._loaded:
            return
        root = Path(__file__).resolve().parent
        models_dir = root / "Models" / "final models"
        metrics_path = models_dir / "crop_pro_metrics.json"
        self._metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

        label_encoder_path = models_dir / "crop_label_encoder.pkl"
        rf_model_path = models_dir / "crop_rf_model.pkl"
        scaler_path = models_dir / "crop_scaler.pkl"
        ann_state_path = models_dir / "crop_ann_model.pth"

        if label_encoder_path.exists():
            self._label_encoder = joblib.load(label_encoder_path)
            self._classes = list(getattr(self._label_encoder, "classes_", []))
        if rf_model_path.exists():
            self._rf_model = joblib.load(rf_model_path)
        if scaler_path.exists():
            self._scaler = joblib.load(scaler_path)
        self._ann_state_path = ann_state_path if ann_state_path.exists() else None

        self._best_model = self._metrics.get("best_model")

        try:
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore

            class CropANN(nn.Module):
                def __init__(self, input_size: int, num_classes: int):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_size, 256),
                        nn.ReLU(),
                        nn.Dropout(0.25),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Dropout(0.20),
                        nn.Linear(128, num_classes),
                    )

                def forward(self, x):  # type: ignore[override]
                    return self.net(x)

            self._torch = torch
            self._CropANN = CropANN
        except Exception:
            self._torch = None
            self._CropANN = None

        self._loaded = True

    def predict(self, N: float, P: float, K: float, ph: float, temperature: float, humidity: float, rainfall: float) -> dict[str, Any]:
        self._load()
        if not (self._label_encoder and self._classes and self._metrics):
            raise RuntimeError("Crop model artifacts are missing.")

        features = {"N": N, "P": P, "K": K, "temperature": temperature, "humidity": humidity, "ph": ph, "rainfall": rainfall}
        feature_order: list[str] = self._metrics.get("features") or ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        features_list = [[float(features[f]) for f in feature_order]]

        best_model = self._best_model or "RandomForest"
        if best_model == "RandomForest":
            if not self._rf_model:
                raise RuntimeError("RandomForest crop model is missing.")
            probabilities = self._rf_model.predict_proba(features_list)[0]
            model_name = "Random Forest (Agent)"
        elif best_model == "ANN":
            if not (self._torch and self._CropANN and self._ann_state_path and self._scaler):
                raise RuntimeError("ANN crop model is missing or PyTorch is not available.")
            transformed = self._scaler.transform(features_list)
            device = self._torch.device("cuda" if self._torch.cuda.is_available() else "cpu")
            model = self._CropANN(input_size=len(feature_order), num_classes=len(self._classes)).to(device)
            model.load_state_dict(self._torch.load(self._ann_state_path, map_location=device))
            model.eval()
            input_tensor = self._torch.tensor(transformed, dtype=self._torch.float32).to(device)
            with self._torch.no_grad():
                outputs = model(input_tensor)
                probabilities = self._torch.softmax(outputs, dim=1)[0].cpu().numpy()
            model_name = "ANN (Agent)"
        else:
            raise RuntimeError(f"Unsupported crop model: {best_model}")

        ranked = sorted(
            (
                {"label": self._classes[index], "probability": round(float(probabilities[index]), 4)}
                for index in range(len(self._classes))
            ),
            key=lambda item: item["probability"],
            reverse=True,
        )
        return {
            "mode": "crop_recommendation",
            "model_name": model_name,
            "label": ranked[0]["label"],
            "top_probability": ranked[0]["probability"]
        }


def get_weather(location: str) -> dict[str, Any]:
    try:
        if "," in location:
            lat_s, lon_s = [part.strip() for part in location.split(",", 1)]
            lat, lon = float(lat_s), float(lon_s)
        else:
            geo_url = "https://geocoding-api.open-meteo.com/v1/search?count=1&name=" + urllib.parse.quote(location)
            with urllib.request.urlopen(geo_url, timeout=15) as resp:
                geo = json.loads(resp.read().decode("utf-8"))
            hit = (geo.get("results") or [None])[0]
            if not hit:
                return {"ok": False, "error": "Location not found"}
            lat, lon = float(hit["latitude"]), float(hit["longitude"])

        forecast_url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
            "&timezone=auto"
        )
        with urllib.request.urlopen(forecast_url, timeout=20) as resp:
            forecast = json.loads(resp.read().decode("utf-8"))
        return {"ok": True, "location": location, "lat": lat, "lon": lon, "daily": forecast.get("daily", {})}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def get_fertilizer_price(name: str, currency: str = "INR") -> dict[str, Any]:
    table_inr = {
        "urea": 7.0,
        "dap": 27.0,
        "mop": 22.0,
        "npk_19_19_19": 30.0,
        "ssp": 8.0,
    }
    key = (name or "").strip().lower().replace(" ", "_")
    price = table_inr.get(key)
    if price is None:
        return {"ok": False, "error": "Unknown fertilizer", "known": sorted(table_inr.keys()), "currency": currency}
    return {"ok": True, "name": key, "price_per_kg": price, "currency": currency}


class FarmAgent:
    def __init__(self) -> None:
        self.chat_history: list[dict[str, str]] = []
        self.context: str = ""
        self._crop_tool = CropModelTool()
        self._api_key = GROQ_API_KEY

    def _llm_available(self) -> bool:
        return bool(self._api_key)

    def status(self) -> dict[str, Any]:
        key = self._api_key or ""
        return {
            "llm_available": self._llm_available(),
            "provider": "groq",
            "has_api_key": bool(self._api_key),
            "api_key_mask": (key[:4] + "..." + key[-4:]) if key else "",
            "api_key_len": len(key),
        }

    def _call_groq(self, system_prompt: str, history: list[dict[str, str]]) -> str:
        if not self._api_key:
            raise RuntimeError("Missing GROQ_API_KEY.")
            
        url = "https://api.groq.com/openai/v1/chat/completions"
        messages = [{"role": "system", "content": system_prompt}] + history
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "temperature": 0.2
        }
        
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "AgronomistAI/1.0"
        }
        resp = _http_json(url, payload=payload, headers=headers)
        
        try:
            return resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return "Error parsing Groq response."

    def _run_react(self, user_message: str, extra_context: Optional[str] = None) -> str:
        sys_prompt = (
            "You are an expert Agronomist AI. You have access to the following tools:\n"
            "1. get_weather(location: str) -> Fetch a short daily weather forecast.\n"
            "2. get_fertilizer_price(name: str) -> Get approximate fertilizer price per kg (e.g., urea, dap).\n"
            "3. run_crop_model(N: float, P: float, K: float, ph: float, temperature: float, humidity: float, rainfall: float) -> Run local crop ML model.\n\n"
            "To use a tool, you MUST output exactly this format:\n"
            "ACTION: tool_name({\"arg1\": \"value1\"})\n"
            "Wait for the system to provide the OBSERVATION. Do NOT output the observation yourself.\n"
            "When you have the final answer, just write the answer directly (no ACTION needed).\n"
        )
        if extra_context:
            sys_prompt += "\nContext:\n" + extra_context.strip()

        # Build a temporary history for the react loop
        loop_history = list(self.chat_history[-10:])
        loop_history.append({"role": "user", "content": user_message})

        tool_handlers = {
            "get_weather": get_weather,
            "get_fertilizer_price": get_fertilizer_price,
            "run_crop_model": self._crop_tool.predict
        }

        for _ in range(5):
            reply = self._call_groq(sys_prompt, loop_history)
            
            # Check if it wants to use a tool
            match = re.search(r"ACTION:\s*([a-zA-Z0-9_]+)\((.*?)\)", reply, re.DOTALL)
            if match:
                func_name = match.group(1)
                args_str = match.group(2)
                
                try:
                    args = json.loads(args_str)
                    if func_name in tool_handlers:
                        result = tool_handlers[func_name](**args)
                    else:
                        result = {"error": f"Unknown tool: {func_name}"}
                except Exception as e:
                    result = {"error": f"Tool execution failed. Did you format the JSON correctly? Error: {str(e)}"}
                
                # Append the assistant's action and the tool's observation
                loop_history.append({"role": "assistant", "content": reply})
                loop_history.append({"role": "user", "content": f"OBSERVATION: {json.dumps(result)}"})
            else:
                return reply.strip()
                
        return "Tool loop limit reached. Please try asking again."

    def generate_action_plan(self, features: dict, crop_result: dict, soil_result: str) -> str:
        self.context = (
            "Farm Context:\n"
            f"- Soil Type: {soil_result}\n"
            f"- Recommended Crop (backend): {crop_result.get('label', 'Unknown')} (Confidence: {crop_result.get('top_probability', 0) * 100:.1f}%)\n"
            f"- Parameters: pH={features.get('soil_ph')}, N={features.get('soil_nitrogen')}, P={features.get('soil_phosphorus')}, K={features.get('soil_potassium')}\n"
            f"- Climate: Temp={features.get('temperature')}C, Rainfall={features.get('rainfall')}mm, Humidity={features.get('humidity')}%\n"
        )
        if not self._llm_available():
            return self._mock_action_plan(features, crop_result, soil_result)

        prompt = (
            "Create a concise Farm Action Plan in Markdown.\n"
            "Include: suitability reasoning (NPK+pH), risks, and 3 actionable steps.\n"
            "Use tools if you need extra data (like weather or fertilizer price).\n"
            "No fluff.\n\n"
            + self.context
        )
        try:
            return self._run_react(prompt, extra_context=self.context)
        except Exception as exc:
            return f"**Agent Error:** {exc}\n\n" + self._mock_action_plan(features, crop_result, soil_result)

    def chat(self, user_message: str) -> str:
        if not self._llm_available():
            return "Mock Mode: add `GROQ_API_KEY` in `.env` to enable the agent."
        try:
            reply = self._run_react(user_message, extra_context=self.context)
            self.chat_history.append({"role": "user", "content": user_message})
            self.chat_history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as exc:
            return f"Sorry—agent error: {exc}"

    def _mock_action_plan(self, features: dict, crop_result: dict, soil_result: str) -> str:
        return (
            "### Agronomist Action Plan (Mock Mode)\n"
            "*Note: Add `GROQ_API_KEY` to the `.env` file to enable the agent.*\n\n"
            "**Assessment:**\n"
            f"Based on pH {features.get('soil_ph')}, N {features.get('soil_nitrogen')}, your environment may fit "
            f"**{str(crop_result.get('label', 'Unknown')).capitalize()}** "
            f"(Confidence: {crop_result.get('top_probability', 0) * 100:.1f}%). Soil type: {soil_result}.\n\n"
            "**Recommendations:**\n"
            "1. **Soil prep:** Till + add compost.\n"
            "2. **NPK tuning:** Adjust nutrients based on soil test.\n"
            "3. **Water plan:** Align irrigation with rainfall.\n"
        )

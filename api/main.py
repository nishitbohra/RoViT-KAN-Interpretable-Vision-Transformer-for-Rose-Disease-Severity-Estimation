"""
FastAPI Application for RoViT-KAN
Main entry point for the BFF layer
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
import io
import base64
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from configs.config import get_config
from models.rovit_kan import RoViTKAN
from data.transforms import InferenceTransform
from explainability.attention_maps import ViTAttentionRollout
from explainability.gradcam import GradCAMPlusPlus

# Initialize FastAPI app
app = FastAPI(
    title="RoViT-KAN API",
    description="Backend-for-Frontend API for Rose Disease Severity Estimation",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global model instance (loaded once at startup)
model: Optional[RoViTKAN] = None
device: Optional[torch.device] = None
config = get_config()

# Class names
CLASS_NAMES = config.data.class_names


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    success: bool
    predicted_class: str
    predicted_class_idx: int
    confidence: float
    class_probabilities: dict
    ordinal_severity: float
    kan_severity: float
    uncertainty_mu: float
    uncertainty_std: float
    inference_time_ms: float
    timestamp: str
    explainability: Optional[dict] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "predicted_class": "Black Spot",
                "predicted_class_idx": 2,
                "confidence": 0.9876,
                "class_probabilities": {
                    "Healthy Leaf": 0.001,
                    "Leaf Holes": 0.008,
                    "Black Spot": 0.9876,
                    "Dry Leaf": 0.0034
                },
                "ordinal_severity": 2.1,
                "kan_severity": 2.15,
                "uncertainty_mu": 2.12,
                "uncertainty_std": 0.23,
                "inference_time_ms": 45.2,
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    timestamp: str


def load_model():
    """Load the trained RoViT-KAN model"""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on device: {device}")
    
    # Initialize model
    model = RoViTKAN(
        embed_dim=config.model.embed_dim,
        hidden_dim=config.model.hidden_dim,
        num_classes=config.data.num_classes,
        kan_layers=config.model.kan_layers,
        kan_num_knots=config.model.kan_num_knots,
        kan_degree=config.model.kan_degree,
        dropout=config.model.dropout,
        pretrained=False  # Loading from checkpoint
    )
    
    # Load checkpoint
    checkpoint_path = config.paths.checkpoints_dir / "best_model.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        print(f"⚠ No checkpoint found at {checkpoint_path}")
        print("⚠ Using randomly initialized weights")
    
    model.to(device)
    model.eval()
    print(f"✓ Model ready on {device}")


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    load_model()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Root endpoint - serves the main UI"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "class_names": CLASS_NAMES
    })


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(device) if device else "none",
        timestamp=datetime.now().isoformat()
    )


async def process_prediction(file: UploadFile) -> PredictionResponse:
    """Core prediction logic shared by both HTML and JSON endpoints"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    allowed_types = ['image/jpeg', 'image/png', 'image/jpg']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )
    
    # Read and preprocess image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Transform
    transform = InferenceTransform(image_size=config.data.image_size)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference with timing
    import time
    start_time = time.time()
    
    with torch.no_grad():
        predictions = model.predict(image_tensor)
    
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Extract predictions
    class_probs = predictions['class_probs'][0].cpu().numpy()
    predicted_idx = int(predictions['class'][0].cpu())
    confidence = float(class_probs[predicted_idx])
    
    # Build class probabilities dict
    probs_dict = {
        CLASS_NAMES[i]: float(class_probs[i]) 
        for i in range(len(CLASS_NAMES))
    }
    
    # Severity and uncertainty
    kan_severity = float(predictions['kan_severity'][0].cpu()) if predictions['kan_severity'] is not None else 0.0
    ordinal_severity = float(predictions['ordinal_severity'][0].cpu()) if predictions['ordinal_severity'] is not None else 0.0
    uncertainty_mu = float(predictions['uncertainty_mu'][0].cpu()) if predictions['uncertainty_mu'] is not None else 0.0
    uncertainty_std = float(predictions['uncertainty_std'][0].cpu()) if predictions['uncertainty_std'] is not None else 0.0
    
    # Generate explainability visualizations
    explainability = await generate_explainability(image_tensor, image, predicted_idx)
    
    return PredictionResponse(
        success=True,
        predicted_class=CLASS_NAMES[predicted_idx],
        predicted_class_idx=predicted_idx,
        confidence=confidence,
        class_probabilities=probs_dict,
        ordinal_severity=ordinal_severity,
        kan_severity=kan_severity,
        uncertainty_mu=uncertainty_mu,
        uncertainty_std=uncertainty_std,
        inference_time_ms=inference_time,
        timestamp=datetime.now().isoformat(),
        explainability=explainability
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_html(request: Request, file: UploadFile = File(...)):
    """
    Predict endpoint that returns HTML for HTMX.
    Returns rendered results template.
    """
    try:
        result = await process_prediction(file)
        return templates.TemplateResponse("results.html", {
            "request": request,
            **result.dict()
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })


async def generate_explainability(image_tensor: torch.Tensor, original_image: Image.Image, class_idx: int) -> dict:
    """Generate attention maps and Grad-CAM++ visualizations"""
    try:
        # Convert PIL to numpy for visualization
        original_np = np.array(original_image.resize((224, 224)))
        
        # Initialize explainability modules
        attention_viz = ViTAttentionRollout(model, device=str(device))
        gradcam = GradCAMPlusPlus(model, device=str(device))
        
        # Generate attention map
        attention_map = attention_viz.generate(image_tensor)
        attention_overlay = attention_viz.overlay_on_image(original_np, attention_map)
        
        # Generate Grad-CAM++
        cam = gradcam.compute(image_tensor, class_idx)
        cam_overlay = gradcam.overlay_on_image(original_np, cam)
        
        # Convert to base64 for JSON response
        def array_to_base64(arr: np.ndarray) -> str:
            import cv2
            _, buffer = cv2.imencode('.png', cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            return base64.b64encode(buffer).decode('utf-8')
        
        return {
            "attention_map": array_to_base64((attention_map * 255).astype(np.uint8)),
            "attention_overlay": array_to_base64(attention_overlay),
            "gradcam": array_to_base64((cam * 255).astype(np.uint8)),
            "gradcam_overlay": array_to_base64(cam_overlay)
        }
    except Exception as e:
        print(f"Explainability generation failed: {e}")
        return None


@app.post("/predict/json")
async def predict_json(file: UploadFile = File(...)):
    """
    Predict endpoint that returns JSON only (for API clients).
    Same as /predict but without the HTML template.
    """
    result = await process_prediction(file)
    return JSONResponse(content=result.dict())


@app.get("/model-info")
async def model_info():
    """Get model information and configuration"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    param_counts = model.count_parameters()
    
    return {
        "model_name": "RoViT-KAN",
        "backbone": config.model.backbone,
        "embed_dim": config.model.embed_dim,
        "num_classes": config.data.num_classes,
        "class_names": CLASS_NAMES,
        "parameters": param_counts,
        "device": str(device),
        "image_size": config.data.image_size
    }


@app.get("/classes")
async def get_classes():
    """Get list of disease classes"""
    return {
        "classes": CLASS_NAMES,
        "severity_map": config.data.severity_map
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

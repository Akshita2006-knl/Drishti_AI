import os
import io
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from google import genai  # Official Google GenAI SDK integration
import uvicorn

app = FastAPI(title="Digital Retinopathy Simplified Full-Stack Server")

# Allow your web frontend to talk to this server securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Path references pointing directly to your local fine-tuned weights folder
TUNED_MODEL_PATH = "./idrid_tuned_model"

print("[INFO] Loading fine-tuned IDRiD optimization matrices from local disk...")
vision_model = CLIPModel.from_pretrained(TUNED_MODEL_PATH)
vision_processor = CLIPProcessor.from_pretrained(TUNED_MODEL_PATH)

# 2. Initialize Google GenAI Client Node (Reads GEMINI_API_KEY from system environment)
print("[INFO] Initializing Google GenAI API Client connection...")
ai_client = genai.Client()

# Mount your current project folder path space to serve assets to the web browser
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def serve_dashboard_ui():
    return FileResponse("index.html")

# Standard clinical diagnostic tiers matching your code vectors
dr_labels = [
    "A normal retinal fundus with no signs of diabetic retinopathy",
    "Mild non-proliferative diabetic retinopathy showing minor microaneurysms",
    "Moderate diabetic retinopathy with noticeable exudates and hemorrhages",
    "Severe retinopathy with multiple cotton wool spots and bleeding",
    "Proliferative diabetic retinopathy with advanced neovascularization"
]

@app.post("/api/diagnose")
async def run_diagnostic_pipeline(file: UploadFile = File(...)):
    try:
        # A. Ingest the user's uploaded image bytes directly into memory
        print(f"[RECEIVE] Processing live user upload: {file.filename}")
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # B. Run Fine-Tuned Local IDRiD Image Diagnostic Grading instantly
        print("[VISION] Grading scan metrics against local tuned neural pathways...")
        inputs = vision_processor(text=dr_labels, images=image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            vision_outputs = vision_model(**inputs)
            
        probabilities = vision_outputs.logits_per_image.softmax(dim=-1).cpu().numpy().flatten()
        best_match_idx = probabilities.argmax()
        
        predicted_condition = dr_labels[best_match_idx]
        predicted_confidence = float(probabilities[best_match_idx])
        
        # Format the matrix probability values array list object structure
        breakdown_matrix = [
            {"condition": label, "confidence": float(prob)}
            for label, prob in zip(dr_labels, probabilities)
        ]
        
        # C. Call official Cloud Gemma model endpoint via GenAI client SDK
        print(f"[GEMMA API] Querying hosted model 'gemma-4-26b-a4b-it' via Google GenAI...")
        gemma_prompt = f"""
        You are a specialist AI Ophthalmic Assistant. Write a professional clinical summary paragraph based on these metrics:
        - Primary Finding: {predicted_condition}
        - Evaluation Certainty Index: {predicted_confidence:.2%}

        Include a single sentence description of what this diagnostic tier means and a recommendation for patient follow-up timing. Keep the response concise, formal, and objective.
        """
        
        response = ai_client.models.generate_content(
            model="gemma-4-26b-a4b-it", # Official cloud-hosted model name string
            contents=gemma_prompt
        )
        
        narrative_report = response.text if response.text else "API connected successfully but no text response returned."
        
        # Return everything to your web application frontend UI dashboard layout
        return {
            "status": "success",
            "prediction": predicted_condition,
            "confidence": predicted_confidence,
            "breakdown": breakdown_matrix,
            "gemma_report": narrative_report
        }
        
    except Exception as e:
        print(f"[PIPELINE ERROR] Execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from contextlib import asynccontextmanager

# Label mappings
LAYER1_ID2LABEL = {0: "non_finance", 1: "finance"}
LAYER2_ID2LABEL = {
    0: "answer_submission",
    1: "clarification_request",
    2: "process_inquiry",
    3: "challenge_assessment",
    4: "off_topic",
    5: "small_talk"
}

# Model paths
LAYER1_MODEL_PATH = "./layer1/layer1_model/checkpoint-516"
LAYER2_MODEL_PATH = "./layer2/layer2_model/checkpoint-1280"

# Global model storage
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Layer 1 model
    print("Loading Layer 1 model...")
    models["layer1_tokenizer"] = AutoTokenizer.from_pretrained(LAYER1_MODEL_PATH)
    models["layer1_model"] = AutoModelForSequenceClassification.from_pretrained(LAYER1_MODEL_PATH)
    models["layer1_model"].to(device)
    models["layer1_model"].eval()
    print("Layer 1 model loaded.")
    
    # Load Layer 2 model
    print("Loading Layer 2 model...")
    models["layer2_tokenizer"] = AutoTokenizer.from_pretrained(LAYER2_MODEL_PATH)
    models["layer2_model"] = AutoModelForSequenceClassification.from_pretrained(LAYER2_MODEL_PATH)
    models["layer2_model"].to(device)
    models["layer2_model"].eval()
    print("Layer 2 model loaded.")
    
    models["device"] = device
    
    yield
    
    # Cleanup
    models.clear()
    print("Models unloaded.")


app = FastAPI(
    title="Classification RAG Guardrail",
    description="Two-layer classification service for finance domain queries",
    version="1.0.0",
    lifespan=lifespan
)


class ClassifyRequest(BaseModel):
    message: str


class LayerResult(BaseModel):
    prediction: str
    confidence: float


class ClassifyResponse(BaseModel):
    layer1: LayerResult
    layer2: LayerResult


def classify_layer1(text: str) -> LayerResult:
    """Run Layer 1 classification (finance vs non-finance)."""
    tokenizer = models["layer1_tokenizer"]
    model = models["layer1_model"]
    device = models["device"]
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
    
    pred_id = probs.argmax().item()
    confidence = probs[pred_id].item()
    
    return LayerResult(
        prediction=LAYER1_ID2LABEL[pred_id],
        confidence=round(confidence, 4)
    )


def classify_layer2(text: str) -> LayerResult:
    """Run Layer 2 classification (intent classification)."""
    tokenizer = models["layer2_tokenizer"]
    model = models["layer2_model"]
    device = models["device"]
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
    
    pred_id = probs.argmax().item()
    confidence = probs[pred_id].item()
    
    return LayerResult(
        prediction=LAYER2_ID2LABEL[pred_id],
        confidence=round(confidence, 4)
    )


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest) -> ClassifyResponse:
    """
    Classify a user message through both layers.
    
    - Layer 1: Determines if the query is finance-related
    - Layer 2: Classifies the intent type
    """
    layer1_result = classify_layer1(request.message)
    layer2_result = classify_layer2(request.message)
    
    return ClassifyResponse(
        layer1=layer1_result,
        layer2=layer2_result
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "models_loaded": len(models) > 0}

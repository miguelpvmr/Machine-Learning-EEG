import asyncio
import threading
import logging
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Domain & Infrastructure Imports
from .infrastructure.virtual_streamer import VirtualStreamer
from .infrastructure.ml_wrappers.sklearn_wrapper import SklearnModelWrapper
from .infrastructure.ui_gateway import WebSocketManager
from .domain.streaming.channel_validator import ChannelValidator
from .domain.streaming.buffer_manager import CircularBufferManager
from .domain.conditioning.butterworth_filter import ButterworthFilter
from .domain.conditioning.notch_filter import NotchFilter
from .domain.conditioning.poly_resampler import PolyResampler
from .domain.conditioning.montage import LongitudinalBipolarMontage
from .domain.conditioning.signal_conditioner_pipe import SignalConditioner
from .domain.extraction.scaler import RobustScaler
from .domain.extraction.spatial_descriptors import SpatialDescriptorCalculator
from .domain.extraction.wavelets import WaveletAnalyzer
from .domain.extraction.engine import FeatureExtractionEngine
from .controllers.stream_controller import StreamController

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Application Initialization
app = FastAPI(title="Real-Time EEG Processing API")
socket_manager = WebSocketManager()

# ============================================================
# COMPOSITION ROOT: DYNAMIC PATH RESOLUTION
# ============================================================

# Resolve project root: app.py -> backend -> src -> Project Root (e.g., D:\TUSZ_project)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "models" / "logit_grid_optimizado_l1" 
TEST_DATA_PATH = BASE_DIR / "tests" / "raw_samples" / "sample_full_recording.parquet"

# Path Validation and Diagnostics
logger.info(f"Root directory detected at: {BASE_DIR}")
if not MODEL_PATH.exists():
    if (MODEL_PATH.with_suffix('.joblib')).exists():
        MODEL_PATH = MODEL_PATH.with_suffix('.joblib')
    else:
        logger.error(f"MODEL ARTIFACT NOT FOUND: {MODEL_PATH}")

if not TEST_DATA_PATH.exists():
    logger.error(f"TEST DATASET NOT FOUND: {TEST_DATA_PATH}")

# ============================================================
# OBJECT GRAPH INITIALIZATION
# ============================================================

# 1. Data Ingestion (Virtual Hardware)
streamer = VirtualStreamer(
    target_fs=250.0, 
    chunk_size_ms=100, 
    parquet_path=str(TEST_DATA_PATH)
)

# 2. Machine Learning & Validation Components
ai_model = SklearnModelWrapper(model_path=str(MODEL_PATH))
validator = ChannelValidator() 
buffer_manager = CircularBufferManager(n_channels=19, fs=250, duration_sec=8)

# 3. Digital Signal Processing (DSP) Pipeline
temporal_filters = [
    ButterworthFilter(lowcut=0.5, highcut=70.0, order=4),
    NotchFilter(freq=60.0),
    PolyResampler(target_fs=250.0)
]
spatial_montage = LongitudinalBipolarMontage()

conditioner = SignalConditioner(
    temporal_pipeline=temporal_filters,
    spatial_transformer=spatial_montage,
    target_window_sec=4.096, 
    target_fs=250.0
)

# 4. Feature Extraction Engine
feature_engine = FeatureExtractionEngine(
    scaler=RobustScaler(),
    spatial_calc=SpatialDescriptorCalculator(),
    wavelet_analyzer=WaveletAnalyzer(wavelet='db4', level=5)
)

# ============================================================
# LIFECYCLE MANAGEMENT & WEBSOCKET ENDPOINTS
# ============================================================

@app.on_event("startup")
async def startup_event():
    """
    Initializes the StreamController in a background thread.
    Captures the main event loop to facilitate thread-safe WebSocket broadcasting.
    """
    main_loop = asyncio.get_running_loop()

    def run_controller_thread():
        # Create a dedicated event loop for the background thread
        thread_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(thread_loop)
        
        def ui_bridge_callback(payload: dict):
            """
            Dispatches processing results to the main loop's WebSocket manager.
            """
            asyncio.run_coroutine_threadsafe(socket_manager.broadcast(payload), main_loop)

        controller = StreamController(
            streamer=streamer,
            validator=validator,
            buffer_manager=buffer_manager,
            signal_conditioner=conditioner,
            feature_engine=feature_engine,
            ai_model=ai_model,
            ui_callback=ui_bridge_callback
        )
        
        logger.info("StreamController initialized and processing signal...")
        controller.start()

    # Execute controller logic in a daemon thread to avoid blocking the API
    threading.Thread(target=run_controller_thread, daemon=True).start()

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    Handles real-time data streaming connections.
    """
    await socket_manager.connect(websocket)
    try:
        while True:
            # Maintain connection and listen for potential client-side heartbeats
            await websocket.receive_text() 
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket)
        logger.info("Client disconnected from WebSocket.")
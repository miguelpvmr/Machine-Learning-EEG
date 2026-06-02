import time
import logging
from typing import Callable

# Relative imports to maintain SOLID structure across packages
from ..infrastructure.eeg_streamer import EEGStreamer
from ..domain.streaming.buffer_manager import CircularBufferManager
from ..domain.streaming.channel_validator import ChannelValidator
from ..domain.conditioning.signal_conditioner_pipe import SignalConditioner
from ..domain.extraction.engine import FeatureExtractionEngine
from ..domain.interfaces.ml_model_interface import IClassificationModel

logger = logging.getLogger(__name__)

class StreamController:
    """
    Orchestrates the real-time EEG processing pipeline.
    
    This controller bridges the Infrastructure and Domain layers, ensuring that 
    data flows from the hardware into the processing pipeline and finally 
    to the UI via a callback mechanism.
    """

    def __init__(
        self,
        streamer: EEGStreamer,                     
        validator: ChannelValidator,                    
        buffer_manager: CircularBufferManager,               
        signal_conditioner: SignalConditioner, 
        feature_engine: FeatureExtractionEngine,
        ai_model: IClassificationModel,  # Using the universal interface
        ui_callback: Callable              
    ):
        self.streamer = streamer
        self.validator = validator
        self.buffer = buffer_manager
        self.signal_conditioner = signal_conditioner
        self.feature_engine = feature_engine
        self.ai_model = ai_model
        self.ui_callback = ui_callback
        
        self.is_running = False
        # Stride logic: Extract features every 1 second based on buffer sampling rate
        self.stride_samples = self.buffer.fs  
        self._accumulated_samples = 0

    def start(self):
        """Initializes hardware connection and starts the main event loop."""
        logger.info("Initializing hardware connection...")
        if not self.streamer.connect():
            logger.error("Failed to connect to hardware. Aborting.")
            return

        self.is_running = True
        logger.info("Stream started. Filling the 8-second buffer...")

        try:
            self._run_loop()
        except KeyboardInterrupt:
            logger.info("Stream interrupted by user.")
        except Exception as e:
            logger.exception(f"Critical error in stream loop: {e}")
        finally:
            self.stop()

    def _run_loop(self):
        """Continuous loop fetching data and managing the sliding window stride."""
        while self.is_running:
            # 1. Fetch raw data from Infrastructure
            raw_channels, raw_data = self.streamer.fetch_chunk()
            
            if raw_data is None or raw_data.size == 0:
                time.sleep(0.01)
                continue

            try:
                # 2. Domain: Validate and map to 10-20 system
                valid_data = self.validator.validate_and_map(raw_channels, raw_data)
            except ValueError as e:
                logger.error(f"Validation failed: {e}")
                self.ui_callback({"status": "error", "message": str(e)})
                self.stop()
                break

            # 3. Domain: Update Circular Buffer (8 seconds total)
            self.buffer.update(valid_data)
            
            # Track samples for the 1-second stride policy
            n_new_samples = valid_data.shape[1]
            self._accumulated_samples += n_new_samples

            # 4. Trigger processing once stride is met and buffer is warm
            if self.buffer.is_ready() and self._accumulated_samples >= self.stride_samples:
                self._process_and_predict()
                self._accumulated_samples -= self.stride_samples

    def _process_and_predict(self):
        """Coordinates DSP and ML Inference."""
        try:
            # A. Extract full 8s window for filtering
            window_8s = self.buffer.get_full_window()
            
            # B. Signal Conditioning (Filters -> Montage -> Extract 4.096s)
            clean_segment = self.signal_conditioner.process(
                raw_8s_buffer=window_8s, 
                original_fs=self.streamer.hardware_fs 
            )
            
            # C. Feature Extraction (Scalers, Wavelets, Spatial Metrics)
            feature_vector = self.feature_engine.extract(clean_segment)
            
            # D. Universal Inference (Returns prediction and probability dict)
            predicted_class, class_probs = self.ai_model.predict(feature_vector)
            
            # E. Update View (UI)
            payload = {
                "status": "ok",
                "prediction": str(predicted_class),
                "probabilities": class_probs,
                "timestamp": time.time()
            }
            self.ui_callback(payload)
            
        except Exception as e:
            logger.error(f"Error during processing/prediction: {e}")
            self.ui_callback({"status": "error", "message": "DSP or Model error."})

    def stop(self):
        """Gracefully shuts down hardware and halts the loop."""
        logger.info("Stopping stream controller...")
        self.is_running = False
        # Ensures safe release of hardware resources
        if hasattr(self.streamer, 'disconnect'):
            self.streamer.disconnect()
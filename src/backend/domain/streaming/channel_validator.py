import numpy as np
from typing import List, Dict

class ChannelValidator:
    """
    Garantiza la integridad espacial de la señal EEG mapeando nombres de hardware 
    a la nomenclatura estándar 10-20 y descartando canales no requeridos.
    """

    def __init__(self):
        # Canales base requeridos para el montaje Bipolar Longitudinal (Double Banana)
        self.required_channels = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
            'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 
            'Fz', 'Cz', 'Pz'
        ]
        
        # Mapeo de normalización para asegurar consistencia (Case Sensitivity)
        self.normalization_map: Dict[str, str] = {
            'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz',
            'F3': 'F3', 'F4': 'F4', 'C3': 'C3', 'C4': 'C4', 'P3': 'P3', 'P4': 'P4',
            'O1': 'O1', 'O2': 'O2', 'F7': 'F7', 'F8': 'F8', 'T3': 'T3', 'T4': 'T4', 
            'T5': 'T5', 'T6': 'T6'
        }
        
        # Sinónimos para compatibilidad con otros datasets o nomenclaturas modernas
        self.synonyms: Dict[str, str] = {
            'T7': 'T3', 'T8': 'T4', 
            'P7': 'T5', 'P8': 'T6'
        }

    def _clean_channel_name(self, raw_name: str) -> str:
        """Limpia prefijos/sufijos de hardware y normaliza el nombre."""
        # Limpieza agresiva de strings
        clean = (raw_name.replace('EEG ', '')
                         .replace('-REF', '')
                         .replace('-LE', '')
                         .strip()
                         .upper())
        
        # 1. Aplicar sinónimos primero (ej. T7 -> T3)
        synonym_clean = self.synonyms.get(clean, clean)
        
        # 2. Normalizar formato (ej. FP1 -> Fp1)
        return self.normalization_map.get(synonym_clean, synonym_clean)

    def validate_and_map(self, raw_channels: List[str], data_packet: np.ndarray) -> np.ndarray:
        """
        Valida la presencia de canales críticos y reordena la matriz de datos.
        
        Args:
            raw_channels: Lista de nombres de canales tal como los entrega el hardware.
            data_packet: Matriz NumPy de forma (n_canales_raw, n_muestras).
            
        Returns:
            np.ndarray: Matriz reordenada de (19, n_muestras) lista para el procesamiento.
        """
        # Limpiar y mapear todos los canales entrantes
        current_map = {self._clean_channel_name(name): i for i, name in enumerate(raw_channels)}
        
        # Verificar canales faltantes
        missing = [ch for ch in self.required_channels if ch not in current_map]
        if missing:
            raise ValueError(f"Falla de Integridad: Faltan canales críticos 10-20: {missing}")
            
        # Extraer índices en el orden exacto de required_channels
        indices = [current_map[ch] for ch in self.required_channels]
        
        # Retornar la matriz filtrada y reordenada (Deep Copy implícito por indexing)
        return data_packet[indices, :]
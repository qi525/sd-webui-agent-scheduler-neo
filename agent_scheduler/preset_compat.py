from typing import Optional


COMPATIBLE_MODEL_PRESETS = {
    "xl": frozenset(
        {
            "WAI-Nsfw-Illustrious-17.safetensors",
            "JANKU v6.9 Final.safetensors",
            "JANKUTrainedNoobaiRouwei_v69.safetensors",
        }
    ),
    "anima": frozenset(
        {
            "anima-base-v1.0.safetensors",
            "anima-turbo-v1.1.safetensors",
            "Anima-2.9B-preview-v1.safetensors",
            "animayume_v15Base.safetensors",
            "oooANIMAV10Anima29B_29B.safetensors",
        }
    ),
}


def infer_compatible_ui_preset(model_filename: str) -> Optional[str]:
    for preset, model_filenames in COMPATIBLE_MODEL_PRESETS.items():
        if model_filename in model_filenames:
            return preset
    return None

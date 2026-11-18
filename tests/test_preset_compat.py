import importlib.util
from pathlib import Path


def load_preset_compat_module():
    path = Path(__file__).resolve().parents[1] / "agent_scheduler" / "preset_compat.py"
    spec = importlib.util.spec_from_file_location("preset_compat_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_known_xl_models_are_classified():
    module = load_preset_compat_module()
    models = [
        "WAI-Nsfw-Illustrious-17.safetensors",
        "JANKU v6.9 Final.safetensors",
        "JANKUTrainedNoobaiRouwei_v69.safetensors",
    ]

    assert all(module.infer_compatible_ui_preset(model) == "xl" for model in models)


def test_known_anima_models_are_classified():
    module = load_preset_compat_module()
    models = [
        "anima-base-v1.0.safetensors",
        "anima-turbo-v1.1.safetensors",
        "Anima-2.9B-preview-v1.safetensors",
        "animayume_v15Base.safetensors",
        "oooANIMAV10Anima29B_29B.safetensors",
    ]

    assert all(module.infer_compatible_ui_preset(model) == "anima" for model in models)


def test_unknown_or_non_exact_model_names_are_not_classified():
    module = load_preset_compat_module()
    models = [
        "wai-nsfw-illustrious-17.safetensors",
        "WAI-Nsfw-Illustrious-17.safetensors [1234567890]",
        "Other-WAI-Nsfw-Illustrious-17.safetensors",
        "anima-base-v1.0.ckpt",
        "unknown-model.safetensors",
    ]

    assert all(module.infer_compatible_ui_preset(model) is None for model in models)

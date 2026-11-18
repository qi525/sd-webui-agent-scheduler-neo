from contextlib import contextmanager
from typing import Any, Callable, Iterator


def resolve_preset_overrides(
    preset: str,
    *,
    available_presets: list[str],
    disabled_presets: list[str],
    preset_values: dict[str, Any],
    valid_dtypes: set[str],
    resolve_checkpoint: Callable[[str], str | None],
) -> dict[str, Any]:
    if not isinstance(preset, str) or preset not in available_presets or preset in disabled_presets:
        raise ValueError(f"Unknown or disabled Forge Preset: {preset!r}")

    overrides: dict[str, Any] = {}
    checkpoint = preset_values.get(f"forge_checkpoint_{preset}")
    if checkpoint:
        if not isinstance(checkpoint, str):
            raise ValueError(f"Invalid checkpoint configured for Forge Preset {preset!r}")
        checkpoint_title = resolve_checkpoint(checkpoint)
        if checkpoint_title is None:
            raise ValueError(f"Checkpoint {checkpoint!r} configured for Forge Preset {preset!r} was not found")
        overrides["sd_model_checkpoint"] = checkpoint_title

    modules = preset_values.get(f"forge_additional_modules_{preset}", [])
    if not isinstance(modules, (list, tuple)) or not all(isinstance(module, str) for module in modules):
        raise ValueError(f"Invalid additional modules configured for Forge Preset {preset!r}")
    overrides["forge_additional_modules"] = list(modules)

    dtype = preset_values.get(f"forge_unet_storage_dtype_{preset}", "Automatic")
    if not isinstance(dtype, str) or dtype not in valid_dtypes:
        raise ValueError(f"Invalid UNet storage dtype {dtype!r} configured for Forge Preset {preset!r}")
    overrides["forge_unet_storage_dtype"] = dtype

    return overrides


def get_preset_overrides(preset: str | None) -> dict[str, Any]:
    if preset is None:
        return {}
    if not isinstance(preset, str):
        raise ValueError(f"Invalid Forge Preset: {preset!r}")
    if not preset.strip():
        return {}

    from modules import sd_models, shared
    from modules_forge import main_entry
    from modules_forge.presets import PresetArch

    prefix = ("forge_checkpoint_", "forge_additional_modules_", "forge_unet_storage_dtype_")
    preset_values = {
        f"{key}{preset}": getattr(shared.opts, f"{key}{preset}", default)
        for key, default in zip(prefix, (None, [], "Automatic"))
    }

    def resolve_checkpoint(name: str) -> str | None:
        checkpoint = sd_models.get_closet_checkpoint_match(name)
        return checkpoint.title if checkpoint is not None else None

    return resolve_preset_overrides(
        preset,
        available_presets=PresetArch.choices(),
        disabled_presets=shared.cmd_opts.disable_preset,
        preset_values=preset_values,
        valid_dtypes=set(main_entry.forge_unet_storage_dtype_options),
        resolve_checkpoint=resolve_checkpoint,
    )


def merge_preset_overrides(
    explicit_overrides: dict[str, Any],
    preset_overrides: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(explicit_overrides)
    for key, value in preset_overrides.items():
        merged.setdefault(key, value)
    return merged


def get_ui_explicit_overrides(named_args: dict[str, Any], checkpoint: str | None) -> dict[str, Any]:
    from modules.infotext_utils import create_override_settings_dict

    explicit = create_override_settings_dict(named_args.get("override_settings_texts") or [])
    from modules import sd_models, shared

    if checkpoint is None or checkpoint == "System":
        explicit["sd_model_checkpoint"] = shared.opts.sd_model_checkpoint
    else:
        checkpoint_info = sd_models.get_closet_checkpoint_match(checkpoint)
        if checkpoint_info is None:
            raise ValueError(f"Checkpoint {checkpoint!r} selected for the queued task was not found")
        explicit["sd_model_checkpoint"] = checkpoint_info.title
    return explicit


@contextmanager
def temporary_ui_preset(preset: str | None, explicit_overrides: dict[str, Any]) -> Iterator[None]:
    preset_overrides = get_preset_overrides(preset)
    runtime_overrides = {
        key: value for key, value in preset_overrides.items() if key not in explicit_overrides
    }
    if not runtime_overrides:
        yield
        return

    from modules import shared, sysinfo
    from modules_forge import main_entry

    original_values = {key: getattr(shared.opts, key) for key in runtime_overrides}
    try:
        sysinfo.set_config(runtime_overrides, is_api=True, run_callbacks=False, save_config=False)
        main_entry.refresh_model_loading_parameters()
        yield
    finally:
        sysinfo.set_config(original_values, is_api=True, run_callbacks=False, save_config=False)
        main_entry.refresh_model_loading_parameters()


def apply_api_preset(
    named_args: dict[str, Any],
    preset: str | None,
) -> dict[str, Any]:
    preset_overrides = get_preset_overrides(preset)
    named_args = dict(named_args)
    named_args["override_settings"] = merge_preset_overrides(
        named_args.get("override_settings") or {},
        preset_overrides,
    )
    named_args["override_settings_restore_afterwards"] = True
    return named_args

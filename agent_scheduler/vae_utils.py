import os
from os import PathLike
from typing import Mapping


def resolve_vae_path(
    vae: str | None,
    installed_vaes: Mapping[str, str | PathLike[str]],
) -> str | None:
    if vae is None or vae in ("", "None", "Automatic"):
        return None
    if not isinstance(vae, str):
        raise ValueError(f"Invalid VAE selection: {vae!r}")

    path = installed_vaes.get(vae)
    if path is None:
        requested_path = os.path.normcase(os.path.abspath(vae))
        path = next(
            (
                installed_path
                for installed_path in installed_vaes.values()
                if os.path.normcase(os.path.abspath(os.fspath(installed_path))) == requested_path
            ),
            None,
        )

    if path is None:
        raise ValueError(f"VAE {vae!r} is not present in the installed VAE list")

    resolved_path = os.fspath(path)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"VAE {vae!r} is listed but its file does not exist: {resolved_path}")

    return resolved_path

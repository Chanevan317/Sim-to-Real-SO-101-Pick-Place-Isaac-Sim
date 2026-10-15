"""Default viewport view of the sim window, saved by pressing V in teleop_so101.py.

The view is stored as eye / lookat points (world frame) in viewport_view.json and applied by
SO101TeleopEnvCfg, so every script that opens a sim window starts with it.
"""

import json
from pathlib import Path

VIEW_PATH = Path(__file__).resolve().parents[1] / "viewport_view.json"


def load_view(path=VIEW_PATH):
    """(eye, lookat) from the saved view, or None if no view was saved."""
    path = Path(path)
    if not path.exists():
        return None
    view = json.loads(path.read_text())
    return tuple(view["eye"]), tuple(view["lookat"])


def save_current_view(path=VIEW_PATH):
    """Save the active viewport camera as eye / lookat. Import only after the app is launched."""
    import omni.usd
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Gf

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("no viewport (running headless?)")
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(viewport.camera_path)
    xf = omni.usd.get_world_transform_matrix(prim)
    eye = xf.ExtractTranslation()
    forward = xf.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()  # cameras look along -Z
    # Kit keeps the orbit centre as "center of interest" (in camera space); fall back to 0.5 m ahead.
    coi = prim.GetAttribute("omni:kit:centerOfInterest")
    distance = abs(coi.Get()[2]) if coi and coi.Get() is not None else 0.5
    lookat = eye + forward * distance
    view = {"eye": [round(v, 4) for v in eye], "lookat": [round(v, 4) for v in lookat]}
    Path(path).write_text(json.dumps(view, indent=2) + "\n")
    return view

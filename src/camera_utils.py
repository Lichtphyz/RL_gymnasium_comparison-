"""
Shared camera-override CLI flags for the video-generating scripts
(record_demo.py, make_progression_video.py, make_grid_video.py). Rather
than hardcode "correct" values (which vary by Gymnasium/MuJoCo version and
would just be guessed), these override whatever the environment's own
DEFAULT_CAMERA_CONFIG already is -- run this to see the real defaults for
your installed version before deciding what to change:

    python3 -c "
    import gymnasium as gym
    env = gym.make('Pusher-v5', render_mode='rgb_array')
    env.reset()
    print(env.unwrapped.mujoco_renderer.default_cam_config)
    "

Rotating 180 degrees = add/subtract 180 from whatever --camera-azimuth
shows as the default. Zooming in = decrease --camera-distance.
"""


def add_camera_args(parser):
    parser.add_argument("--camera-distance", type=float, default=None,
                         help="Camera distance from the lookat point. Smaller = zoomed in. "
                              "Omit to use the environment's own default.")
    parser.add_argument("--camera-azimuth", type=float, default=None,
                         help="Camera azimuth in degrees. Add/subtract 180 from the environment's "
                              "default to view from the opposite side.")
    parser.add_argument("--camera-elevation", type=float, default=None,
                         help="Camera elevation in degrees (typically negative, looking down).")
    parser.add_argument("--camera-lookat", type=str, default=None,
                         help="Comma-separated x,y,z point the camera looks at, e.g. '0,0,0'.")


def build_camera_config(args) -> dict:
    """Returns a dict suitable for gym.make(..., default_camera_config=...),
    containing only the fields the user actually overrode -- fields left
    as None are omitted, so the environment's own defaults fill them in."""
    config = {}
    if args.camera_distance is not None:
        config["distance"] = args.camera_distance
    if args.camera_azimuth is not None:
        config["azimuth"] = args.camera_azimuth
    if args.camera_elevation is not None:
        config["elevation"] = args.camera_elevation
    if args.camera_lookat is not None:
        import numpy as np
        config["lookat"] = np.array([float(v) for v in args.camera_lookat.split(",")])
    return config

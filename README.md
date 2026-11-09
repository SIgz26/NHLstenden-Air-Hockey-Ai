# NHLstenden-Air-Hockey-Ai
NHL Stenden project for developing and training AI agents in an air hockey simulation using reinforcement learning.

## Orion Vision on Ubuntu

Install Python's virtual-environment support and the Qt libraries used by the
desktop interface:

```bash
sudo apt update
sudo apt install python3-venv libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
  libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 \
  libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0 \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

From the `Orion Vision` directory, install dependencies and start the app with
the project-managed environment:

```bash
uv sync
uv run main.py
```

Alternatively, use a virtual environment with pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

Do not run the standalone Python executable managed by `uv` directly; it does
not automatically use the project's installed dependencies.

The `requirements.txt` file includes the application, training, analysis, and
test dependencies. Using Allied Vision cameras additionally requires installing
the vendor's Vimba X SDK; `vmbpy` alone does not include the camera runtime.

If no USB camera is listed, check that Linux exposes a device under `/dev/video*`
and that your user can access it. On Ubuntu, camera access is commonly granted
through the `video` group; after adding your user to that group, log out and
back in. GigE cameras require the Vimba X runtime and its transport layers in
addition to the Python package.

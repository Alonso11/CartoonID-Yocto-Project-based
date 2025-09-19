#!/bin/bash
# Activar micromamba environment for GStreamer + YOLO
eval "$(micromamba shell hook --shell bash)"
micromamba activate gstreamer-yolo

# Get Python version and paths
PYTHON_VERSION=$(python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
SITE_PACKAGES="$HOME/micromamba/envs/gstreamer-yolo/lib/$PYTHON_VERSION/site-packages"
GST_PLUGINS_PATH="$HOME/micromamba/envs/gstreamer-yolo/lib/gstreamer-1.0"

# Set environment variables (use current directory)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GST_PLUGIN_PATH="$GST_PLUGINS_PATH:$PROJECT_DIR/sources/plugins"
export GST_PLUGIN_SYSTEM_PATH="$GST_PLUGINS_PATH"
export PYTHONPATH="$PROJECT_DIR/sources/plugins:$SITE_PACKAGES"
export LD_LIBRARY_PATH="$HOME/micromamba/envs/gstreamer-yolo/lib:$LD_LIBRARY_PATH"

# Intel-specific optimizations
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export KMP_AFFINITY=granularity=fine,compact,1,0

# Información del ambiente
echo "=== Ambiente GStreamer+YOLO Activado ==="
echo "Python: $(which python)"
echo "Python version: $PYTHON_VERSION"
echo "GST_PLUGIN_PATH: $GST_PLUGIN_PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "=========================="

# Verify GStreamer Python plugin
echo "Checking GStreamer Python plugin..."
gst-inspect-1.0 python

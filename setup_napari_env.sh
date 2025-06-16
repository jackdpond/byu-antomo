#!/bin/bash

# exit on error
set -e

# Default environment name
conda_name="napari-env"
# Default yaml file
env_yaml="napari-env.yaml"

# Parse command line arguments
while getopts "n:y:" opt; do
  case $opt in
    n) conda_name=$OPTARG ;;
    y) env_yaml=$OPTARG ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done

# check if conda is installed
if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not installed. Please install conda first."
    exit 1
fi

# build conda environment
if ! conda env create -n "$conda_name" -f "$env_yaml"; then
    echo "Failed to create conda environment. Exiting environment setup."
    exit 1
else
    echo "Conda environment $conda_name created successfully."
fi

# Set up alias for antomo in the environment
CONDA_ENV_PATH=$(conda info --base)/envs/$conda_name
mkdir -p "$CONDA_ENV_PATH/etc/conda/activate.d"
echo "alias antomo='ipython -i ~/groups/fslg_imagseg/jackson/Napari/antomo.py'" > "$CONDA_ENV_PATH/etc/conda/activate.d/alias_antomo.sh"
chmod +x "$CONDA_ENV_PATH/etc/conda/activate.d/alias_antomo.sh"

# Print activation instructions
cat <<EOF

To activate your environment, run:
  conda activate $conda_name
EOF

# Add auto-activation to .bashrc (or .zshrc if you use zsh)
SHELL_RC="$HOME/.bashrc"
if [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
fi

NAPARI_DIR="$HOME/groups/fslg_imagseg/jackson/Napari"
ENV_NAME="$conda_name"

AUTO_ACTIVATE_BLOCK="
# >>> auto-activate napari-env when entering $NAPARI_DIR >>>
function cd() {
    builtin cd \"\$@\"
    if [ \"\$PWD\" = \"$NAPARI_DIR\" ]; then
        conda activate $ENV_NAME
    fi
}
# <<< auto-activate napari-env <<<
"

# Only add if not already present
if ! grep -q 'auto-activate napari-env' \"$SHELL_RC\"; then
    echo \"\$AUTO_ACTIVATE_BLOCK\" >> \"$SHELL_RC\"
    echo \"[INFO] Added auto-activation block to \$SHELL_RC\"
else
    echo \"[INFO] Auto-activation block already present in \$SHELL_RC\"
fi 
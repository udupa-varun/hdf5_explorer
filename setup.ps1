# exit on first failure
$ErrorActionPreference = "Stop"

Write-Output "Beginning setup..."

# where was the script called from?
$INVOKE_DIR = (Get-Item .).FullName
Write-Output "Invoke dir: $INVOKE_DIR"

# where is the script located?
$SCRIPT_DIR = $PSScriptRoot
Write-Output "Script dir: $SCRIPT_DIR"
Set-Location $SCRIPT_DIR

$PYVER = "3.10"
$PYCMD = "py -", $PYVER -join ""

# check if python is installed
Write-Output "Looking for Python $PYVER..."
$CMD = $PYCMD, "--version" -join " "
Invoke-Expression $CMD
Write-Output "Valid Python version detected."

# name for the venv
$VENV_NAME = "env"

# check if venv already exists
$VENV_PATH = Join-Path -Path $SCRIPT_DIR -ChildPath $VENV_NAME
if (Test-Path -Path $VENV_PATH) {
    Write-Output "An existing virtual environment '$VENV_NAME' was detected. It will be cleared and recreated."
}

# create new venv
Write-Output "Creating new virtual environment '$VENV_NAME'..."
$CMD = $PYCMD, "-m", "venv", $VENV_NAME, "--clear" -join " "
Invoke-Expression $CMD

# activate venv
Write-Output "Activating virtual environment '$VENV_NAME'..."
$CMD = ".", $VENV_NAME, "Scripts", "Activate.ps1" -join "\"
Invoke-Expression $CMD

# upgrade pip
Write-Output "Upgrading pip..."
$CMD = "python -m pip install --upgrade pip"
Invoke-Expression $CMD

# install requirements
Write-Output "Installing required packages inside virtual environment. This could take a few minutes..."
# install wheel first
$CMD = "pip install wheel"
Invoke-Expression $CMD
# install app requirements
$CMD = "pip install -r .\requirements.txt"
Invoke-Expression $CMD

# reset location
Set-Location $INVOKE_DIR
Write-Output "Setup is complete. Run 'launch_app.ps1' to launch the app."

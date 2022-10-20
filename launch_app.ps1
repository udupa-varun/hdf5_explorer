# exit on first failure
$ErrorActionPreference = "Stop"

Write-Output "Launching app..."
Write-Output "Hit Ctrl+C to quit at any time."

# where was the script called from?
$INVOKE_DIR = (Get-Item .).FullName
Write-Output "Invoke dir: $INVOKE_DIR"

# where is the script located?
$SCRIPT_DIR = $PSScriptRoot
Write-Output "Script dir: $SCRIPT_DIR"
Set-Location $SCRIPT_DIR

# name for the venv
$VENV_NAME = "env"

# activate venv
Write-Output "Activating virtual environment '$VENV_NAME'..."
$CMD = ".", $VENV_NAME, "Scripts", "Activate.ps1" -join "\"
Invoke-Expression $CMD

# Display python version
$CMD = "python --version"
Invoke-Expression $CMD

# Run app
$CMD = "streamlit run .\app\app.py"
Invoke-Expression $CMD

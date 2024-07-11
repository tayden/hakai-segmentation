@echo off
setlocal enabledelayedexpansion

:: Check if pip is installed
python -m pip --version >nul 2>&1
if !errorlevel! neq 0 (
    echo Pip is not installed. Installing pip...
    python -m ensurepip --default-pip
    if !errorlevel! neq 0 (
        echo Failed to install pip. Please install it manually.
        exit /b 1
    )
    echo Pip has been installed successfully.
) else (
    echo Pip is already installed.
)

:: Check if uv is installed
uv --version >nul 2>&1
if !errorlevel! neq 0 (
    echo uv is not installed. Installing uv...
    pip install uv
    if !errorlevel! neq 0 (
        echo Failed to install uv. Please install it manually.
        exit /b 1
    )
    echo uv has been installed successfully.
) else (
    echo uv is already installed.
)

:: Check if "kom" virtual environment exists
if exist "kom" (
    echo Virtual environment "kom" already exists. Activating...
) else (
    echo Creating virtual environment "kom"...
    uv venv kom
)

:: Activate the virtual environment
call kom\Scripts\activate.bat
:: Check if activation was successful by verifying the VIRTUAL_ENV variable
if not defined VIRTUAL_ENV (
    echo Failed to activate the virtual environment. Exiting.
    exit /b 1
)
echo Virtual environment "kom" is now active.

:: Check for CUDA-capable GPU and determine CUDA version
set "cuda_version="
call nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    call nvidia-smi --query-gpu=driver_version --format=csv,noheader > temp_driver_version.txt
    set /p driver_version=<temp_driver_version.txt
    del temp_driver_version.txt
    echo CUDA-capable GPU detected with driver version !driver_version!

    :: Determine CUDA version based on driver version
    if !driver_version! geq 525.60 (
        set "cuda_version=12.1"
    ) else if !driver_version! geq 450.80 (
        set "cuda_version=11.8"
    )

    echo !cuda_version! is the recommended CUDA version for your GPU driver version.

    if defined cuda_version (
        if !cuda_version!==12.1 (
            echo Installing PyTorch and torchvision with CUDA 12.1 support...
            uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        ) else if !cuda_version!==11.8 (
            echo Installing PyTorch and torchvision with CUDA 11.8 support...
            uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        )
    ) else (
        echo Your GPU driver version !driver_version! is not compatible with CUDA 11.8 or 12.1. Installing CPU version...
        uv pip install torch torchvision
    )
) else (
    echo No CUDA-capable GPU detected or CUDA is not installed. Installing PyTorch and torchvision for CPU...
    uv pip install torch torchvision
)

:: Install kelp-o-matic
echo Installing kelp-o-matic...
uv pip install .
:: kelp-o-matic

echo Python package installation script completed successfully.

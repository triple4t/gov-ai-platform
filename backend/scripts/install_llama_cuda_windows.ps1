# Rebuild llama-cpp-python with NVIDIA CUDA on Windows.
# Upstream publishes Linux CUDA wheels only for recent versions; Windows needs a local compile.
#
# Prerequisites:
#   - CUDA Toolkit installed (nvcc), e.g. "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"
#   - Visual Studio 2022 C++ build tools (Desktop development with C++)
#   - Stop uvicorn / any Python process using llama-cpp so DLLs are not locked
#
# Usage (from repo root or backend):
#   cd backend
#   .\scripts\install_llama_cuda_windows.ps1

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $here

$py = Join-Path $here "venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error ('venv not found at ' + $here + '\venv - create it and pip install -r requirements.txt first.')
}

Write-Host "Stopping pip/uvicorn is recommended if a previous uninstall left DLL locks." -ForegroundColor Yellow

# Clean broken partial uninstall (tilde-prefixed folder from pip on Windows)
$site = Join-Path $here "venv\Lib\site-packages"
Get-ChildItem $site -Directory -Filter "~lama*" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "Removing stale folder: $($_.FullName)"
    Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
}

$env:CMAKE_ARGS = "-DGGML_CUDA=on"
# Help CMake find CUDA if not on PATH in some shells
if (-not $env:CUDA_PATH) {
    $cudaRoot = "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $cudaRoot) {
        $latest = Get-ChildItem $cudaRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1
        if ($latest) {
            $env:CUDA_PATH = $latest.FullName
            Write-Host "Set CUDA_PATH=$($env:CUDA_PATH)"
        }
    }
}

& (Join-Path $here "venv\Scripts\pip.exe") uninstall llama-cpp-python -y
& (Join-Path $here "venv\Scripts\pip.exe") install llama-cpp-python==0.3.19 --force-reinstall --no-cache-dir

Write-Host 'Verify GPU offload:' -ForegroundColor Green
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$verifyPy = Join-Path $scriptDir 'verify_llama_gpu.py'
& $py $verifyPy

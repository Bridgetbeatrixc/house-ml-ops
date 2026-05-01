$ErrorActionPreference = "Stop"

Set-Location "C:\Users\bridg\OneDrive\Desktop\house-ml-ops"

$kaggleToken = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
if (-not (Test-Path $kaggleToken)) {
    throw "Kaggle token not found at $kaggleToken. Download kaggle.json from Kaggle Account > Create New API Token, then place it there."
}

New-Item -ItemType Directory -Force -Path "data" | Out-Null
python -m kaggle competitions download -c house-prices-advanced-regression-techniques -p data --force

$zipPath = "data\house-prices-advanced-regression-techniques.zip"
if (Test-Path $zipPath) {
    Expand-Archive -Path $zipPath -DestinationPath "data" -Force
}

if (-not (Test-Path "data\train.csv")) {
    throw "Download finished, but data\train.csv was not found. Make sure you accepted the Kaggle competition rules."
}

Write-Host "Dataset ready at data\train.csv"

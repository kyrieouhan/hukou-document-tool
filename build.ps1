$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = Join-Path $root '.venv/Scripts/python.exe'
$dist = Join-Path $root 'dist/1.2.0'
$work = Join-Path $root 'build/1.2.0'

Push-Location -LiteralPath $root
try {
    if (-not (Test-Path -LiteralPath $python)) { throw 'Create .venv and install requirements.txt first.' }
    & $python 'scripts/fetch_models.py' --check
    if ($LASTEXITCODE -ne 0) { throw 'Run scripts/fetch_models.py to prepare the model files.' }
    New-Item -ItemType Directory -Path $dist,$work -Force | Out-Null

    $originalPath = $env:PATH
    try {
        $basePython = & $python -c 'import sys; print(sys.base_prefix)'
        # Prefer operating-system DLLs over unrelated copies elsewhere on PATH.
        $env:PATH = "$(Split-Path $python);$basePython;$env:SystemRoot\System32;$env:SystemRoot"
        & $python -m PyInstaller --clean --noconfirm --windowed --onedir --name HukouTool `
            --distpath $dist --workpath $work --specpath $work --paths source `
            --collect-data rapidocr_onnxruntime --collect-binaries onnxruntime `
            --add-data "$root/source/hukou/ocr_models;hukou/ocr_models" `
            --version-file "$root/docs/windows_version.txt" `
            --exclude-module tkinter --exclude-module matplotlib --exclude-module pandas 'source/main.py'
        if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed.' }
    }
    finally { $env:PATH = $originalPath }

    Get-Item -LiteralPath (Join-Path $dist 'HukouTool/HukouTool.exe')
}
finally { Pop-Location }

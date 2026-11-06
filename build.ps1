# Builds zero on Windows with MSVC (Visual Studio Build Tools, "Desktop development with C++").
# Usage: .\build.ps1            -> bin\zero.exe (command line) and bin\zero-gui.exe (window)
#        .\build.ps1 -Only cli  |  -Only gui
# The compiler is found with vswhere, then the usual folders; $env:ZERO_VCVARS = a vcvars64.bat path overrides it.
param([string]$Only = "")
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

function Find-VcVars {
  if ($env:ZERO_VCVARS -and (Test-Path $env:ZERO_VCVARS)) { return $env:ZERO_VCVARS }
  $pf86 = [Environment]::GetFolderPath('ProgramFilesX86')
  $vswhere = Join-Path $pf86 "Microsoft Visual Studio\Installer\vswhere.exe"
  if ($pf86 -and (Test-Path $vswhere)) {
    $paths = & $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    foreach ($p in $paths) {
      $v = Join-Path $p "VC\Auxiliary\Build\vcvars64.bat"
      if (Test-Path $v) { return $v }
    }
  }
  foreach ($base in @([Environment]::GetFolderPath('ProgramFilesX86'), $env:ProgramFiles)) {
    if (-not $base) { continue }
    $hits = Get-ChildItem -Path (Join-Path $base "Microsoft Visual Studio") -Recurse -Filter vcvars64.bat -ErrorAction SilentlyContinue |
      Sort-Object FullName -Descending
    if ($hits) { return $hits[0].FullName }
  }
  if (Get-Command cl -ErrorAction SilentlyContinue) { return "" }  # (a Developer prompt: cl already set up)
  return $null
}

$vcvars = Find-VcVars
if ($null -eq $vcvars) {
  throw "No Visual Studio C++ compiler found: install Visual Studio Build Tools with 'Desktop development with C++', or set `$env:ZERO_VCVARS to a vcvars64.bat."
}

# the kernels, embedded into the exe as a byte array
& (Join-Path $root "scripts\embed_kernels.ps1") (Join-Path $root "kernels\zero.cl") (Join-Path $root "src\kernels_gen.h")

$out = Join-Path $root "bin"
New-Item -ItemType Directory -Force $out | Out-Null
$pre = if ($vcvars) { "`"$vcvars`" >nul && " } else { "" }
# /MT: the C++ runtime inside the exe, so a PC without the Visual C++ Redistributable runs it; /utf-8: UTF-8 sources;
# the manifest makes the process's code page UTF-8 (paths with any characters) and lifts the 260-character limit.
$common = "cl /nologo /utf-8 /std:c++20 /O2 /Oi /EHsc /MT /W4 /permissive- /Zc:__cplusplus /D_CRT_SECURE_NO_WARNINGS /DNOMINMAX /I`"$root\src`" /Fo`"$out\\`""
$manifest = "/MANIFEST:EMBED /MANIFESTINPUT:`"$root\tools\utf8.manifest`""

if ($Only -ne "gui") {
  Write-Host "building bin\zero.exe"
  cmd /c "$pre$common /Fe`"$out\zero.exe`" `"$root\src\main.cpp`" /link $manifest"
  if ($LASTEXITCODE -ne 0) { throw "build failed: zero.exe" }
}
if ($Only -ne "cli") {
  Write-Host "building bin\zero-gui.exe"
  if (-not (Test-Path (Join-Path $out "zero.exe"))) { throw "build the command line first: it goes inside the installer" }
  cmd /c "$pre rc /nologo /i `"$root\tools`" /i `"$out`" /fo `"$out\zero.res`" `"$root\tools\zero.rc`""
  if ($LASTEXITCODE -ne 0) { throw "build failed: zero.rc" }
  cmd /c "$pre$common /DZERO_GUI /Fe`"$out\zero-gui.exe`" `"$root\src\main.cpp`" /link /SUBSYSTEM:WINDOWS $manifest user32.lib gdi32.lib shell32.lib ole32.lib advapi32.lib dwmapi.lib d2d1.lib dwrite.lib uuid.lib `"$out\zero.res`""
  if ($LASTEXITCODE -ne 0) { throw "build failed: zero-gui.exe" }
  # the installer is this same program under another name: it holds the command line and installs itself
  Copy-Item -Force (Join-Path $out "zero-gui.exe") (Join-Path $out "zero-setup.exe")
}
Remove-Item (Join-Path $out "*.obj") -ErrorAction SilentlyContinue
Remove-Item (Join-Path $out "*.res") -ErrorAction SilentlyContinue
Write-Host "done"

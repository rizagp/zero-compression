# Round-trip test: compresses a folder (or files) with the GPU and with the CPU, lists and tests the archives,
# extracts each with both codecs, and compares every extracted file with the original (SHA-256).
# Usage: .\tests\roundtrip.ps1 -Source <folder or file> [-Work <temp folder>] [-Block 19]
param(
  [Parameter(Mandatory = $true)][string]$Source,
  [string]$Work = (Join-Path $env:TEMP "zero-roundtrip"),
  [int]$Block = 19
)
$ErrorActionPreference = 'Stop'
$zero = Join-Path $PSScriptRoot "..\bin\zero.exe"
if (-not (Test-Path $zero)) { throw "build first: .\build.ps1" }
New-Item -ItemType Directory -Force $Work | Out-Null
Remove-Item (Join-Path $Work "*") -Recurse -Force -ErrorAction SilentlyContinue

function Hashes($root) {
  $root = (Resolve-Path $root).Path
  $h = @{}
  Get-ChildItem $root -Recurse -File | ForEach-Object {
    $rel = $_.FullName.Substring($root.Length).TrimStart('\')
    $h[$rel] = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
  }
  return $h
}
function Compare-Trees($a, $b, $label) {
  $ha = Hashes $a
  $hb = Hashes $b
  $bad = 0
  foreach ($k in $ha.Keys) { if (-not $hb.ContainsKey($k)) { Write-Host "  MISSING in $label : $k"; $bad++ } elseif ($ha[$k] -ne $hb[$k]) { Write-Host "  DIFFERENT in $label : $k"; $bad++ } }
  foreach ($k in $hb.Keys) { if (-not $ha.ContainsKey($k)) { Write-Host "  EXTRA in $label : $k"; $bad++ } }
  if ($bad -eq 0) { Write-Host "  $label : $($ha.Count) files identical" -ForegroundColor Green } else { Write-Host "  $label : $bad problems" -ForegroundColor Red }
  return $bad
}

$in = (Resolve-Path $Source).Path
$name = Split-Path $in -Leaf
$orig = if (Test-Path $in -PathType Container) { $in } else { Split-Path $in -Parent }
$problems = 0
foreach ($codec in @("gpu", "cpu")) {
  $flag = @(); if ($codec -eq "cpu") { $flag += "--cpu" }
  $arc = Join-Path $Work "$name-$codec.zero"
  Write-Host "== compress ($codec) ==" -ForegroundColor Cyan
  & $zero c $in -o $arc --block $Block @flag -q
  if ($LASTEXITCODE -ne 0) { throw "compress failed ($codec)" }
  Write-Host "== list ==" -ForegroundColor Cyan
  (& $zero l $arc | Select-Object -First 3) | ForEach-Object { Write-Host $_ }
  foreach ($dcodec in @("gpu", "cpu")) {
    $dflag = @(); if ($dcodec -eq "cpu") { $dflag += "--cpu" }
    Write-Host "== test ($codec archive, $dcodec decoder) ==" -ForegroundColor Cyan
    & $zero t $arc @dflag -q
    if ($LASTEXITCODE -ne 0) { throw "test failed" }
    $out = Join-Path $Work "x-$codec-$dcodec"
    Write-Host "== extract ($codec archive, $dcodec decoder) ==" -ForegroundColor Cyan
    & $zero x $arc -o $out @dflag -q
    if ($LASTEXITCODE -ne 0) { throw "extract failed" }
    if (Test-Path $in -PathType Container) { $problems += Compare-Trees $in (Join-Path $out $name) "$codec->$dcodec" }
    else { $problems += Compare-Trees $orig $out "$codec->$dcodec" }
  }
}
if ($problems -eq 0) { Write-Host "ALL OK" -ForegroundColor Green } else { Write-Host "$problems PROBLEMS" -ForegroundColor Red; exit 1 }

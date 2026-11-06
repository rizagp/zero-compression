# Installs with bin\zero-setup.exe --silent, checks what it put where, compresses and extracts with the installed copies,
# uninstalls, and checks nothing is left. It refuses to run over a real installation, and leaves the Explorer entries as
# it found them.
param([string]$Work = "$env:TEMP\zero-installer-test")
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$dir = "$env:LOCALAPPDATA\Programs\zero"
$lnk = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\zero.lnk"
$un = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\zero'
$failed = 0
function Check([string]$what, [bool]$ok) { "$(if ($ok) { 'ok    ' } else { 'FAILED' }) $what"; if (-not $ok) { $script:failed++ } }
function Open-Command { (Get-Item -LiteralPath 'HKCU:\Software\Classes\zero.Archive\shell\open\command' -ErrorAction SilentlyContinue)?.GetValue('') }
if (Test-Path $dir) { throw "zero is installed in ${dir}: uninstall it first" }
if (Get-Process zero-gui -ErrorAction SilentlyContinue) { throw 'close the zero window first' }
$pathBefore = [Environment]::GetEnvironmentVariable('Path', 'User')
$menuBefore = Open-Command
[void][IO.Directory]::CreateDirectory($Work)
[IO.File]::WriteAllText("$Work\a.txt", ('hello installer ' * 5000))
try {
  $p = Start-Process "$root\bin\zero-setup.exe" -ArgumentList '--silent --add-path' -PassThru -Wait
  Check 'the installer exits with success' ($p.ExitCode -eq 0)
  Check 'the two programs are in place' ((Test-Path "$dir\zero.exe") -and (Test-Path "$dir\zero-gui.exe"))
  Check 'a Start menu entry exists' (Test-Path $lnk)
  $u = Get-ItemProperty -LiteralPath $un -ErrorAction SilentlyContinue
  Check 'it is listed in Settings, Apps' ($u -and $u.DisplayName -eq 'zero' -and $u.UninstallString -like '*zero-gui.exe*--uninstall*')
  Check 'the Explorer entries name the installed copy' ((Open-Command) -like "*$dir\zero-gui.exe*")
  Check 'PATH has the folder once' (([Environment]::GetEnvironmentVariable('Path', 'User') -split ';' | Where-Object { $_ -eq $dir }).Count -eq 1)
  & "$dir\zero.exe" c "$Work\a.txt" -o "$Work\a.zero" -q 2>$null | Out-Null
  & "$dir\zero.exe" x "$Work\a.zero" -o "$Work\x" -q 2>$null | Out-Null
  Check 'the installed command line round-trips a file' ((Get-FileHash "$Work\a.txt").Hash -eq (Get-FileHash "$Work\x\a.txt").Hash)
  $p = Start-Process "$root\bin\zero-setup.exe" -ArgumentList '--silent --add-path' -PassThru -Wait
  Check 'installing again updates in place' ($p.ExitCode -eq 0 -and ([Environment]::GetEnvironmentVariable('Path', 'User') -split ';' | Where-Object { $_ -eq $dir }).Count -eq 1)
  $p = Start-Process "$root\bin\zero-gui.exe" -ArgumentList '--uninstall --silent' -PassThru -Wait
  Check 'a copy that is not the installed one refuses to uninstall' ($p.ExitCode -eq 1 -and (Test-Path "$dir\zero.exe"))
  $p = Start-Process "$dir\zero-gui.exe" -ArgumentList '--uninstall --silent' -PassThru -Wait
  $t = Get-Date
  while ((Test-Path $dir) -and ((Get-Date) - $t).TotalSeconds -lt 15) { Start-Sleep -Milliseconds 200 }
  Check 'uninstalling removes the folder' (-not (Test-Path $dir))
  Check 'and the Start menu entry and the Apps entry' ((-not (Test-Path $lnk)) -and (-not (Test-Path -LiteralPath $un)))
  Check 'and the Explorer entries' (-not (Test-Path -LiteralPath 'HKCU:\Software\Classes\zero.Archive'))
  Check 'and PATH is as it was' ([Environment]::GetEnvironmentVariable('Path', 'User') -eq $pathBefore)
  Check 'the archives are still there' (Test-Path "$Work\a.zero")
} finally {
  if ($menuBefore -and (Open-Command) -ne $menuBefore) { Start-Process "$root\bin\zero-gui.exe" -ArgumentList '--install-shell' -Wait }
  [IO.Directory]::Delete($Work, $true)
}
if ($failed) { "FAILURES: $failed"; exit 1 }
'ALL OK'

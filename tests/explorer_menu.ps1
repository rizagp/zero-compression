# What Explorer does with the entries zero-gui.exe registers: the verbs of a file, a folder and an archive; several
# files gathering in one window; opening an archive as a double-click does; extracting; removing the entries. It leaves
# them as it found them. Windows open on the desktop for a moment.
param([string]$Work = "$env:TEMP\zero-explorer-test")
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$gui = "$root\bin\zero-gui.exe"; $cli = "$root\bin\zero.exe"
if (-not ('Z' -as [type])) { Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices;
public static class Z {
  [DllImport("user32.dll")] public static extern IntPtr SendMessageW(IntPtr h, uint m, IntPtr w, IntPtr l);
  [DllImport("user32.dll")] public static extern bool PostMessageW(IntPtr h, uint m, IntPtr w, IntPtr l);
  [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern IntPtr FindWindowW(string c, string t);
}
'@ }
$WM_TEST = 0x8000 + 20; $failed = 0
function Check([string]$what, [bool]$ok) { "$(if ($ok) { 'ok    ' } else { 'FAILED' }) $what"; if (-not $ok) { $script:failed++ } }
function Wait-Window([int]$ms = 8000) { $t = Get-Date; do { $h = [Z]::FindWindowW('ZeroMainWindow', 'zero'); if ($h -ne 0) { return $h }; Start-Sleep -Milliseconds 50 } while (((Get-Date) - $t).TotalMilliseconds -lt $ms); return [IntPtr]::Zero }
function Close-Window { $h = [Z]::FindWindowW('ZeroMainWindow', 'zero'); if ($h -ne 0) { [void][Z]::PostMessageW($h, 0x10, [IntPtr]::Zero, [IntPtr]::Zero) }; $t = Get-Date; while ((Get-Process zero-gui -ErrorAction SilentlyContinue) -and ((Get-Date) - $t).TotalSeconds -lt 8) { Start-Sleep -Milliseconds 100 } }
function Query([IntPtr]$h, [int]$what) { [int64][Z]::SendMessageW($h, $WM_TEST, [IntPtr]$what, [IntPtr]::Zero) }
if (Get-Process zero-gui -ErrorAction SilentlyContinue) { throw 'close the zero window first' }
$was = Test-Path -LiteralPath 'HKCU:\Software\Classes\zero.Archive'
[void][IO.Directory]::CreateDirectory("$Work\sub")
foreach ($n in 'a.txt', 'b.txt', 'sub\c.txt') { [IO.File]::WriteAllText("$Work\$n", ("hello $n " * 2000)) }
& $cli c "$Work\a.txt" "$Work\sub" -o "$Work\arc.zero" -q 2>$null | Out-Null
try {
  Start-Process $gui -ArgumentList '--install-shell' -Wait
  $app = New-Object -ComObject Shell.Application
  $ns = $app.Namespace($Work)
  $names = @($ns.ParseName('a.txt').Verbs() | ForEach-Object { $_.Name })
  Check 'a file has "Compress with zero"' ($names -contains 'Compress with zero')
  $names = @($ns.ParseName('sub').Verbs() | ForEach-Object { $_.Name })
  Check 'a folder has "Compress with zero"' ($names -contains 'Compress with zero')
  $names = @($ns.ParseName('arc.zero').Verbs() | ForEach-Object { $_.Name })
  Check 'an archive has "Extract here" and "Open with zero"' (($names -contains 'Extract here') -and ($names -contains 'Open with zero'))
  Check 'an archive does not offer to compress it' ($names -notcontains 'Compress with zero')

  $ns.ParseName('a.txt').InvokeVerb('zero.compress')
  $h = Wait-Window; Start-Sleep -Milliseconds 400
  Check 'the verb opens the new archive screen with the file' ((Query $h 3) -eq 1 -and (Query $h 4) -eq 1)
  Close-Window

  $ps = foreach ($n in 'a.txt', 'b.txt', 'sub') { Start-Process $gui -ArgumentList "--add `"$Work\$n`"" -PassThru }
  $h = Wait-Window; Start-Sleep -Milliseconds 1500
  Check 'three at once gather in one window' (@(Get-Process zero-gui).Count -eq 1 -and (Query $h 4) -eq 3)
  Close-Window

  Start-Process "$Work\arc.zero"
  $h = Wait-Window; Start-Sleep -Milliseconds 400
  Check 'opening an archive, as a double-click does, shows its contents' ($h -ne 0 -and (Query $h 3) -eq 2)
  Close-Window

  $ns.ParseName('arc.zero').InvokeVerb('zero.extract')
  $t = Get-Date
  do { Start-Sleep -Milliseconds 100 } until ((Test-Path "$Work\arc\a.txt") -or ((Get-Date) - $t).TotalSeconds -gt 20)
  Start-Sleep -Milliseconds 800
  $same = (Test-Path "$Work\arc\a.txt") -and (Test-Path "$Work\arc\sub\c.txt") -and ((Get-FileHash "$Work\a.txt").Hash -eq (Get-FileHash "$Work\arc\a.txt").Hash) -and ((Get-FileHash "$Work\sub\c.txt").Hash -eq (Get-FileHash "$Work\arc\sub\c.txt").Hash)
  Check 'Extract here writes the files into a folder beside the archive' $same
  Close-Window

  Start-Process $gui -ArgumentList '--uninstall-shell' -Wait
  $left = @('HKCU:\Software\Classes\.zero', 'HKCU:\Software\Classes\zero.Archive', 'HKCU:\Software\Classes\*\shell\zero.compress', 'HKCU:\Software\Classes\Directory\shell\zero.compress') | Where-Object { Test-Path -LiteralPath $_ }
  Check 'removing leaves nothing in the registry' (@($left).Count -eq 0)
} finally {
  Close-Window
  if ($was) { Start-Process $gui -ArgumentList '--install-shell' -Wait } else { Start-Process $gui -ArgumentList '--uninstall-shell' -Wait }
  [IO.Directory]::Delete($Work, $true)
}
if ($failed) { "FAILURES: $failed"; exit 1 }
'ALL OK'

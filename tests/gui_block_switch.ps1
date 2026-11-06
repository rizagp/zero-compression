# Drives zero-gui.exe through what a person does: several archives in one window, changing the block size between
# them, so the kernels are rebuilt inside one process. Each archive is then tested and extracted by the CLI.
param(
  [Parameter(Mandatory = $true)][string[]]$Source, # one or more files of at least some MiB
  [string]$Work = "$env:TEMP\zero-gui-test",
  [string]$Gui = "$PSScriptRoot\..\bin\zero-gui.exe",
  [string]$Cli = "$PSScriptRoot\..\bin\zero.exe",
  [int[]]$Blocks = @(19, 20, 16, 20, 19),          # log2 block size, in order
  [int]$TimeoutSeconds = 300
)
$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class W {
  [StructLayout(LayoutKind.Sequential)] struct CopyData { public IntPtr dwData; public int cbData; public IntPtr lpData; }
  [DllImport("user32.dll")] public static extern IntPtr SendMessageW(IntPtr h, uint m, IntPtr w, IntPtr l);
  [DllImport("user32.dll")] public static extern bool PostMessageW(IntPtr h, uint m, IntPtr w, IntPtr l);
  public static void SetArchive(IntPtr h, string path) {
    IntPtr s = Marshal.StringToHGlobalUni(path);
    var c = new CopyData { dwData = (IntPtr)1, cbData = (path.Length + 1) * 2, lpData = s };
    IntPtr p = Marshal.AllocHGlobal(Marshal.SizeOf(c));
    Marshal.StructureToPtr(c, p, false);
    SendMessageW(h, 0x4A, IntPtr.Zero, p);
    Marshal.FreeHGlobal(p);
    Marshal.FreeHGlobal(s);
  }
}
'@
$ID_COMPRESS = 110; $WM_COMMAND = 0x111; $WM_CLOSE = 0x10; $WM_TEST = 0x8000 + 20
[void][IO.Directory]::CreateDirectory($Work)
$want = @{}
foreach ($s in $Source) { $want[(Split-Path $s -Leaf)] = (Get-FileHash $s).Hash }
$p = Start-Process -FilePath $Gui -ArgumentList (($Source | ForEach-Object { "`"$_`"" }) -join ' ') -PassThru
$failed = 0
try {
  $t0 = Get-Date
  while ($p.MainWindowHandle -eq 0 -and ((Get-Date) - $t0).TotalSeconds -lt 20) { Start-Sleep -Milliseconds 200; $p.Refresh() }
  $w = $p.MainWindowHandle
  if ($w -eq 0) { throw 'the window did not open' }
  $n = 0
  foreach ($bl in $Blocks) {
    ++$n
    $arc = Join-Path $Work "a$n.zero"
    [W]::SetArchive($w, $arc)
    [void][W]::SendMessageW($w, $WM_TEST, [IntPtr]1, [IntPtr]($bl - 16))
    [void][W]::PostMessageW($w, $WM_COMMAND, [IntPtr]$ID_COMPRESS, [IntPtr]::Zero)
    Start-Sleep -Milliseconds 300
    $t0 = Get-Date
    while (([int64][W]::SendMessageW($w, $WM_TEST, [IntPtr]2, [IntPtr]::Zero)) -eq 1) {
      if ($p.HasExited) { throw "the window exited with code $($p.ExitCode)" }
      if (((Get-Date) - $t0).TotalSeconds -gt $TimeoutSeconds) { throw 'timed out' }
      Start-Sleep -Milliseconds 100
    }
    $secs = ((Get-Date) - $t0).TotalSeconds
    if (([int64][W]::SendMessageW($w, $WM_TEST, [IntPtr]2, [IntPtr]::Zero)) -eq 2) {
      "block log $bl : FAILED"
      ++$failed
      continue
    }
    $x = Join-Path $Work "x$n"
    & $Cli t $arc -q | Out-Null
    $tc = $LASTEXITCODE
    & $Cli x $arc -o $x -q | Out-Null
    $xc = $LASTEXITCODE
    $same = $xc -eq 0
    if ($same) { foreach ($k in $want.Keys) { if ((Get-FileHash (Join-Path $x $k)).Hash -ne $want[$k]) { $same = $false } } }
    if ($tc -eq 0 -and $same) { "block log $bl : identical, {0:n1} s" -f $secs }
    else { "block log $bl : FAILED: test $tc, extract $xc"; ++$failed }
    [IO.Directory]::Delete($x, $true)
    [IO.File]::Delete($arc)
  }
  [void][W]::PostMessageW($w, $WM_CLOSE, [IntPtr]::Zero, [IntPtr]::Zero)
  if (-not $p.WaitForExit(15000)) { "the window did not close"; ++$failed }
  elseif ($p.ExitCode -ne 0) { "the window closed with code 0x{0:x}" -f $p.ExitCode; ++$failed }
} finally {
  if (-not $p.HasExited) { $p.Kill() }
  [IO.Directory]::Delete($Work, $true)
}
if ($failed) { "FAILURES: $failed"; exit 1 }
'ALL OK'

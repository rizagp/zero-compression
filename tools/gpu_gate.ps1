# Is the GPU free? A GPU run here must not overlap another project's: one process's lost device can take down the others,
# and two at once spoil both sets of numbers. Exits 0 when free, 1 when busy, and says what is using it.
# -Stable S: free the whole time for S seconds (a benchmark leaves idle gaps between its runs: one quiet moment means
# nothing). -Wait N: keep trying for up to N seconds.
param([int]$Wait = 0, [int]$Stable = 20, [double]$Threshold = 8)
function Busy {
  $found = @()
  # programs of the other project, whatever they are doing: a loaded server holds the GPU's memory
  $found += Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*arcinferzero*' } | ForEach-Object { "$($_.ProcessName) (arcinferzero)" }
  # any program driving a GPU engine; the desktop's own drawing is small and does not count
  $peak = @{}
  foreach ($i in 1..3) {
    try {
      (Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction Stop).CounterSamples | ForEach-Object {
        $m = [regex]::Match($_.InstanceName, 'pid_(\d+).*engtype_(\w+)')
        if ($m.Success -and $_.CookedValue -gt [double]($peak[$m.Groups[1].Value])) { $peak[$m.Groups[1].Value] = $_.CookedValue }
      }
    } catch { }
    Start-Sleep -Milliseconds 300
  }
  $desktop = 'dwm', 'csrss', 'explorer', 'chrome', 'msedge', 'firefox', 'Code', 'claude', 'Claude', 'SearchHost', 'ShellExperienceHost', 'StartMenuExperienceHost', 'TextInputHost'
  foreach ($k in $peak.Keys) {
    if ($peak[$k] -lt $Threshold) { continue }
    $n = (Get-Process -Id ([int]$k) -ErrorAction SilentlyContinue).ProcessName
    if ($n -and $n -notin $desktop) { $found += "$n (GPU {0:n0} %)" -f $peak[$k] }
  }
  return $found
}
$t0 = Get-Date
$quiet = Get-Date
$last = ''
while ($true) {
  $b = @(Busy)
  if ($b.Count) { $quiet = Get-Date; $last = $b -join ', ' }
  elseif (((Get-Date) - $quiet).TotalSeconds -ge $Stable) { "GPU free for $Stable s"; exit 0 }
  if (((Get-Date) - $t0).TotalSeconds -ge [Math]::Max($Wait, $Stable + 10)) { "GPU busy: $last"; exit 1 }
  Start-Sleep -Seconds 3
}

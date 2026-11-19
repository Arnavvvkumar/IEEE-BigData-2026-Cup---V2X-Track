# Final walker-capable CARLA capture during the agreed GPU window (see coordination event claude_carla_gpu_window_plan).
# DirectX off-screen rendering device, no CUDA. Walker probe first; full capture only if walkers return points.
# A watchdog kills CARLA at once if any OpenPCDet / CUDA compute process appears. CARLA is always stopped at the end.
$ErrorActionPreference = 'Stop'
$O = 'C:\Users\arnav\Downloads\BIG DATA IEEE\experiments\claude_carla_overnight'
$RT = 'C:\Users\arnav\Downloads\BIG DATA IEEE\experiments\roadside_pilot\runtime'
$PY = 'C:\Users\arnav\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = "$RT\stock-python;$RT\collector-windows-deps"
$exe = "$RT\stock-candidate\CarlaUE4\Binaries\Win64\CarlaUE4-Win64-Shipping.exe"
$log = "$O\runtime\capture_final.log"
function Note($m) { $line = "$((Get-Date).ToUniversalTime().ToString('o')) $m"; Add-Content $log $line; Write-Output $line }
function StopCarla { Get-Process | Where-Object { $_.Name -match '^CarlaUE4' } | Stop-Process -Force -Confirm:$false -ErrorAction SilentlyContinue; Start-Sleep 5 }

$state = (wsl -e bash "${BDC_ROOT}/experiments/claude_carla_overnight/tools/window_state.sh") -join ' '
if ($state -ne 'READY') { Note "ABORT: window not ready: $state"; exit 3 }
StopCarla
Note 'launching CARLA DirectX offscreen on NVIDIA, Town02, low quality'
$p = Start-Process -FilePath $exe -ArgumentList 'CarlaUE4', '/Game/Carla/Maps/Town02', '-RenderOffScreen', '-graphicsadapter=0', '-quality-level=Low', '-nosound', '-carla-rpc-port=2200', '-carla-streaming-port=0', '-ResX=320', '-ResY=240' -PassThru -WindowStyle Hidden -RedirectStandardOutput "$O\runtime\carla-final.stdout.log" -RedirectStandardError "$O\runtime\carla-final.stderr.log"
$p.Id | Out-File "$O\runtime\carla.pid" -Encoding ascii
# watchdog job: kill CARLA if OpenPCDet or CUDA compute processes appear
$wd = Start-Job -ArgumentList $p.Id -ScriptBlock {
  param($cid)
  while (Get-Process -Id $cid -ErrorAction SilentlyContinue) {
    $s = (wsl -e bash "${BDC_ROOT}/experiments/claude_carla_overnight/tools/busy_state.sh") -join ' '
    if ($s -match '^BUSY') { Get-Process | Where-Object { $_.Name -match '^CarlaUE4' } | Stop-Process -Force; "WATCHDOG KILL: $s"; return }
    Start-Sleep 10
  }
}
try {
  & $PY "$O\tools\wait_server.py" 2200 300 | Out-Null
  if ($LASTEXITCODE -ne 0) { throw 'server not ready' }
  Note "server ready; NVIDIA used MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  $mode = @()
  & $PY "$O\carla_gen.py" --out "$O\probe_v4_render" --clips 1 --frames 10 --step 3 --warmup 20 --sem-every 1 --seed 77000 | Out-Null
  $ped = (& $PY "$O\tools\sem_tags.py" "$O\probe_v4_render\clip_000" | Select-String 'ped-tag points (\d+)' | ForEach-Object { [int]$_.Matches[0].Groups[1].Value } | Measure-Object -Sum).Sum
  Note "probe no_rendering_mode=True pedestrian semantic points: $ped"
  if ($ped -eq 0) {
    & $PY "$O\carla_gen.py" --out "$O\probe_v5_render_world" --clips 1 --frames 10 --step 3 --warmup 20 --sem-every 1 --seed 77000 --world-rendering | Out-Null
    $ped = (& $PY "$O\tools\sem_tags.py" "$O\probe_v5_render_world\clip_000" | Select-String 'ped-tag points (\d+)' | ForEach-Object { [int]$_.Matches[0].Groups[1].Value } | Measure-Object -Sum).Sum
    Note "probe world rendering pedestrian semantic points: $ped"
    if ($ped -eq 0) { throw 'walkers invisible even with a render device' }
    $mode = @('--world-rendering')
  }
  Note "full capture start mode=$($mode -join ' ')"
  & $PY "$O\carla_gen.py" --out "$O\native_v1" --clips 25 --frames 40 --step 3 --warmup 30 --sem-every 4 --seed 91000 @mode *>> "$O\runtime\capture_final_gen.log"
  if ($LASTEXITCODE -ne 0) { throw "carla_gen exit $LASTEXITCODE" }
  Note "full capture done: $((Get-ChildItem "$O\native_v1" -Directory).Count) clips"
}
catch { Note "ERROR: $_" }
finally {
  StopCarla
  Note "CARLA stopped; NVIDIA used MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader); watchdog: $(Receive-Job $wd -ErrorAction SilentlyContinue)"
  Stop-Job $wd -ErrorAction SilentlyContinue; Remove-Job $wd -Force -ErrorAction SilentlyContinue
}

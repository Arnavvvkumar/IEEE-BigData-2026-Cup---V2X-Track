# CPU-only (-nullrhi) CARLA v3 capture. usage: carla_v3_capture.ps1 -Out native_v3_test -Town Town02 -Clips 2 -Seed 93000 [-Extra '--tilt-deg','6.47']
param([string]$Out, [string]$Town = 'Town02', [int]$Clips = 2, [int]$Seed = 93000, [int]$Frames = 40, [string[]]$Extra = @())
$O = 'C:\Users\arnav\Downloads\BIG DATA IEEE\experiments\claude_carla_overnight'
$RT = 'C:\Users\arnav\Downloads\BIG DATA IEEE\experiments\roadside_pilot\runtime'
$PY = 'C:\Users\arnav\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = "$RT\stock-python;$RT\collector-windows-deps"
$exe = "$RT\stock-candidate\CarlaUE4\Binaries\Win64\CarlaUE4-Win64-Shipping.exe"
if (-not (Get-Process | Where-Object { $_.Name -match '^CarlaUE4' })) {
  $p = Start-Process -FilePath $exe -ArgumentList 'CarlaUE4', '/Game/Carla/Maps/Town02', '-nullrhi', '-nosound', '-carla-rpc-port=2200', '-carla-streaming-port=0' -PassThru -WindowStyle Hidden
  $p.PriorityClass = 'BelowNormal'
}
& $PY "$O\tools\wait_server.py" 2200 300 | Out-Null
& $PY "$O\carla_gen_v3.py" --out "$O\$Out" --town $Town --clips $Clips --frames $Frames --step 3 --warmup 30 --sem-every 1000 --seed $Seed @Extra

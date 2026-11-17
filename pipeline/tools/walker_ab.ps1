# CPU-only (-nullrhi) walker-visibility A/B: default invincibility with AI controllers vs direct WalkerControl.
param([string]$Server = 'nullrhi')
$O = 'C:\Users\arnav\Downloads\BIG DATA IEEE\experiments\claude_carla_overnight'
$RT = 'C:\Users\arnav\Downloads\BIG DATA IEEE\experiments\roadside_pilot\runtime'
$PY = 'C:\Users\arnav\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = "$RT\stock-python;$RT\collector-windows-deps"
$exe = "$RT\stock-candidate\CarlaUE4\Binaries\Win64\CarlaUE4-Win64-Shipping.exe"
Get-Process | Where-Object { $_.Name -match '^CarlaUE4' } | Stop-Process -Force -ErrorAction SilentlyContinue; Start-Sleep 4
$p = Start-Process -FilePath $exe -ArgumentList 'CarlaUE4', '/Game/Carla/Maps/Town02', '-nullrhi', '-nosound', '-carla-rpc-port=2200', '-carla-streaming-port=0' -PassThru -WindowStyle Hidden
$p.PriorityClass = 'BelowNormal'
& $PY "$O\tools\wait_server.py" 2200 200 | Out-Null
foreach ($v in @(@('ab_ai_invincible', '--walker-mode', 'ai'), @('ab_direct_invincible', '--walker-mode', 'direct'))) {
  $name = $v[0]; $rest = $v[1..($v.Length - 1)]
  if (Test-Path "$O\$name") { Rename-Item "$O\$name" "$name.old$(Get-Random)" }
  & $PY "$O\carla_gen.py" --out "$O\$name" --clips 1 --frames 10 --step 3 --warmup 20 --sem-every 1 --seed 77000 @rest | Out-Null
  $s = & $PY "$O\tools\sem_tags.py" "$O\$name\clip_000" | Select-String 'ped-tag'
  "$name : $s"
}

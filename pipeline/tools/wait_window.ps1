$sh = "${BDC_ROOT}/experiments/claude_carla_overnight/tools/window_state.sh"
for ($i = 0; $i -lt 400; $i++) {
  $r = (wsl -e bash $sh) -join ' '
  if ($r -eq 'READY') { "READY at $((Get-Date).ToUniversalTime().ToString('o'))"; exit 0 }
  if ($r -match 'DENIED') { "DENIED $r"; exit 2 }
  Start-Sleep 30
}
"TIMEOUT $r"

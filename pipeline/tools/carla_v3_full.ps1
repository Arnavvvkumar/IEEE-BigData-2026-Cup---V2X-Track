# Full v3 capture: 4 towns x 30 clips x 40 scans = 4,800 scans, CPU only.
$t = 'C:\Users\arnav\Downloads\BIG DATA IEEE\experiments\claude_carla_overnight\tools\carla_v3_capture.ps1'
$i = 0
foreach ($town in @('Town02', 'Town01', 'Town03', 'Town05')) {
  & $t -Out "native_v3_$town" -Town $town -Clips 30 -Seed (94000 + 1000 * $i) -Frames 40
  $i++
}
'V3_CAPTURE_DONE'

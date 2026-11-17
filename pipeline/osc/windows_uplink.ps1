$dest = Join-Path $env:TEMP "up8.bin"
$stream = [System.IO.File]::Create($dest)
$stream.SetLength(8000000)
$stream.Close()
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& curl.exe -s -o NUL -X POST --data-binary "@$dest" --max-time 90 "https://speed.cloudflare.com/__up"
$code = $LASTEXITCODE
$sw.Stop()
if ($code -eq 0) {
  $mbps = 8 / $sw.Elapsed.TotalSeconds
  Write-Output ("windows_MBps {0:N3} seconds {1:N1}" -f $mbps, $sw.Elapsed.TotalSeconds)
} else {
  Write-Output "windows probe failed exit $code"
}
Remove-Item $dest -ErrorAction SilentlyContinue

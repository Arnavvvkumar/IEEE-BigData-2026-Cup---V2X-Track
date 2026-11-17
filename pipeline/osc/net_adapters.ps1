Get-NetAdapter | Select-Object Name, Status, LinkSpeed, MacAddress | Format-Table -AutoSize | Out-String -Width 200
Write-Output "--- tools ---"
foreach ($name in @("ssh", "scp", "rsync", "curl")) {
  $cmd = Get-Command $name -ErrorAction SilentlyContinue
  if ($cmd) { Write-Output "$name $($cmd.Source)" } else { Write-Output "$name missing" }
}

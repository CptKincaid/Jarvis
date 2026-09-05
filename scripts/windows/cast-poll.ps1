# Jarvis cast poller. Runs as Hunter at logon. Talks to the Spark only.
#
# Copy to %USERPROFILE%\jarvis\cast-poll.ps1 and put the phone key in
# %USERPROFILE%\jarvis\cast-token.txt (readable only by him).
#
# IT RECEIVES A VERB FROM A CLOSED SET. It never receives, builds or runs a
# command string. Every command line below is a literal in this file, and
# that is the line that keeps this from being remote execution into his
# live session. jarvis/webapp.py enforces the same set at the other end.
#
# What it sends: the INDEX of the monitor holding the mouse pointer, and the
# x-offset and width of each monitor. No cursor coordinate, no window
# handle, no title, no process name and no path leaves this machine.
$ErrorActionPreference = 'SilentlyContinue'
Add-Type -AssemblyName System.Windows.Forms

$Spark    = '192.168.50.109'
$Port     = 8765
$Token    = (Get-Content "$env:USERPROFILE\jarvis\cast-token.txt" -Raw).Trim()
$RustDesk = "$env:ProgramFiles\RustDesk\rustdesk.exe"   # FIXED
$Target   = '192.168.50.109'                            # FIXED

$seq = -1
$castPid = 0

function Get-MonIndex {
  $p = [System.Windows.Forms.Cursor]::Position
  $all = [System.Windows.Forms.Screen]::AllScreens
  for ($i = 0; $i -lt $all.Count; $i++) {
    if ($all[$i].Bounds.Contains($p)) { return $i }
  }
  return -1
}

function Get-Layout {
  (([System.Windows.Forms.Screen]::AllScreens |
    ForEach-Object { "$($_.Bounds.X),$($_.Bounds.Width)" }) -join ',')
}

while ($true) {
  try {
    $body = @{ mon = (Get-MonIndex); layout = (Get-Layout); seq = $seq } |
            ConvertTo-Json -Compress
    # The Spark holds the request for up to 25 s and answers the instant a
    # verb is set: about 2.4 requests a minute against its 60/minute limit,
    # with a cast landing in well under a second.
    $r = Invoke-RestMethod -Method Post -TimeoutSec 30 `
           -Uri "http://${Spark}:${Port}/api/cast" `
           -Headers @{ Authorization = "Bearer $Token" } `
           -ContentType 'application/json' -Body $body
    [int]$newSeq = $r.seq
    [string]$verb = $r.verb
    if ($newSeq -ne $seq) {
      $seq = $newSeq
      if ($verb -eq 'show-spark') {
        if ($castPid -ne 0) { Stop-Process -Id $castPid -Force }
        $castPid = (Start-Process -FilePath $RustDesk `
                     -ArgumentList '--connect', $Target -PassThru).Id
      }
      elseif ($verb -eq 'stop') {
        if ($castPid -ne 0) { Stop-Process -Id $castPid -Force; $castPid = 0 }
      }
      # anything else: do nothing, deliberately. No else branch acts.
    }
  } catch { Start-Sleep -Seconds 5 }
  Start-Sleep -Milliseconds 200
}

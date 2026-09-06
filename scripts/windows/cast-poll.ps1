# Jarvis cast poller. Runs as Hunter at logon. Talks to the Spark only.
#
# Copy to %USERPROFILE%\jarvis\cast-poll.ps1 and put the phone key in
# %USERPROFILE%\jarvis\cast-token.txt (readable only by him). Installing it
# is copying this file; there is nothing to run and nothing to answer.
#
# IT RECEIVES A VERB FROM A CLOSED SET. It never receives, builds or runs a
# command string. Every command line below is a literal in this file, and
# that is the line that keeps this from being remote execution into his
# live session. jarvis/webapp.py enforces the same set at the other end.
#
# What it sends: the INDEX of the monitor holding the mouse pointer, the
# x-offset and width of each monitor, the sequence it has ACTED ON, and --
# new in round 3 -- a failure CODE from its own closed set when it could
# not act. No cursor coordinate, no window handle, no title, no process
# name, no path and no Windows error string leaves this machine.
#
# THE RECEIPT IS THE POINT OF THIS FILE, AND IT WAS WRONG. Jarvis reads
# $seq back as proof that the cast landed. Round 2 committed it BEFORE
# Start-Process, under $ErrorActionPreference = 'SilentlyContinue', so
# RustDesk not being installed at the fixed path -- the ordinary way this
# fails -- still produced a receipt, and Jarvis said "The Spark's screen is
# on HPCOMPUTER, sir." with no window anywhere (MEASURED: 1 launch attempt,
# 0 windows up). Now $seq moves only after a launch this script has WATCHED
# survive, and a launch that fails reports a code instead.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms

$Spark    = '192.168.50.109'
$Port     = 8765
$TokenFile = "$env:USERPROFILE\jarvis\cast-token.txt"
$RustDesk = "$env:ProgramFiles\RustDesk\rustdesk.exe"   # FIXED
$Target   = '192.168.50.109'                            # FIXED

# How long to watch a freshly started viewer before believing in it. The
# same idea as castview.VIEWER_SETTLE_S at the other end: a viewer that
# cannot start dies almost at once.
$SettleMs = 700

$seq = -1
$castPid = 0
$fail = ''
$failSeq = -1

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

function Stop-Cast {
  param([int]$ProcId)
  if ($ProcId -ne 0) {
    try { Stop-Process -Id $ProcId -Force -ErrorAction Stop } catch { }
  }
  return 0
}

# Start the viewer and WATCH IT. Returns the pid, or 0 with $script:fail
# set to one of the codes castview.HELPER_FAILS lists. The catch block
# reads $_ and throws NONE of it back at the Spark: the code is chosen
# here, from the closed set, and the Windows message stays on Windows.
function Start-Viewer {
  if (-not (Test-Path -LiteralPath $RustDesk)) {
    $script:fail = 'no-viewer'
    return 0
  }
  try {
    $p = Start-Process -FilePath $RustDesk `
           -ArgumentList '--connect', $Target -PassThru -ErrorAction Stop
  } catch {
    $script:fail = 'launch-failed'
    return 0
  }
  if ($null -eq $p) {
    $script:fail = 'launch-failed'
    return 0
  }
  Start-Sleep -Milliseconds $SettleMs
  $p.Refresh()
  if ($p.HasExited) {
    $script:fail = 'viewer-exited'
    return 0
  }
  $script:fail = ''
  return $p.Id
}

while ($true) {
  try {
    # The key is read each round, inside the try. $ErrorActionPreference is
    # 'Stop' now -- silently swallowing errors is what this round is fixing
    # -- so reading it once at the top would kill the whole poller at logon
    # if the file were not there yet, and he would have nothing to look at.
    # Read here, it simply retries, and he can rotate the key without
    # signing out. A poll runs about 2.4 times a minute.
    $Token = (Get-Content $TokenFile -Raw).Trim()
    $body = @{ mon = (Get-MonIndex); layout = (Get-Layout); seq = $seq;
               fail = $fail; failseq = $failSeq } | ConvertTo-Json -Compress
    # The Spark holds the request for up to 25 s and answers the instant a
    # verb is set: about 2.4 requests a minute against its 60/minute limit,
    # with a cast landing in well under a second.
    $r = Invoke-RestMethod -Method Post -TimeoutSec 30 `
           -Uri "http://${Spark}:${Port}/api/cast" `
           -Headers @{ Authorization = "Bearer $Token" } `
           -ContentType 'application/json' -Body $body
    [int]$newSeq = $r.seq
    [string]$verb = $r.verb
    # A sequence this script has ALREADY FAILED is never retried: it would
    # relaunch on every round for as long as Jarvis held the verb. Jarvis
    # gets the code, gives up, and sets a new sequence.
    if (($newSeq -ne $seq) -and ($newSeq -ne $failSeq)) {
      $ok = $true
      if ($verb -eq 'show-spark') {
        $castPid = Stop-Cast -ProcId $castPid
        $castPid = Start-Viewer
        $ok = ($castPid -ne 0)
      }
      elseif ($verb -eq 'stop') {
        $castPid = Stop-Cast -ProcId $castPid
      }
      # anything else: do nothing, deliberately. No else branch acts.
      if ($ok) {
        # ONLY HERE. The receipt says "I have this one and I did it".
        $seq = $newSeq
        $fail = ''
        $failSeq = -1
      }
      else {
        $failSeq = $newSeq
      }
    }
  } catch { Start-Sleep -Seconds 5 }
  Start-Sleep -Milliseconds 200
}

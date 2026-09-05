# HPCOMPUTER: give Jarvis a STANDARD user for the file link, not an admin one.
#
# Why (measured 2026-09-04): the Spark's key ~/.ssh/hpcomputer sat in
# C:\ProgramData\ssh\administrators_authorized_keys, so the key the Jarvis
# process holds was an unrestricted Administrator login on the Windows PC.
# The file lane (jarvis/tools/remote.py) only ever copies one file into one
# inbox folder and lists/copies out of allow-listed folders; it needs none of
# that. After this script the same key logs in as a local user "jarvis" that
# is in Users only, key-only (no password login), with:
#   modify  on  C:\Users\<owner>\jarvis-inbox     (pushes land here)
#   read    on  C:\Users\<owner>\jarvis-outbox    (pulls come from here)
#   read    on  C:\Users\<owner>\Desktop          ("from HPCOMPUTER's desktop")
#   read    on  C:\Users\<owner>\Downloads        ("from HPCOMPUTER's downloads")
# and nothing else. Revoke Desktop/Downloads with the icacls lines at the
# bottom if pulls from the outbox alone are enough.
#
# The user's authorized_keys lives OUTSIDE its profile
# (C:\ProgramData\ssh\jarvis_authorized_keys, named by a Match User block in
# sshd_config), because a brand-new local user has no profile directory until
# it first logs on, and sshd would otherwise find no key file.
#
# RUN AS ADMINISTRATOR on HPCOMPUTER (or over the admin ssh link, once):
#   powershell -ExecutionPolicy Bypass -File hpcomputer_standard_user.ps1 `
#       -Owner h2pey -PublicKey "ssh-ed25519 AAAA... spark"
# The password is generated here, never printed, never needed: the account
# is key-only. An admin can reset it any time with Set-LocalUser.
#
# ORDER OF OPERATIONS matters and the Spark drives it:
#   1. run this (creates the user; the admin key still works)
#   2. from the Spark: ssh -i ~/.ssh/hpcomputer jarvis@<host> whoami  -> hpcomputer\jarvis
#   3. ONLY THEN remove the Spark's line from administrators_authorized_keys
#      (see -RemoveAdminKey below), and set remote.user = "jarvis" in
#      ~/.config/jarvis/assistant.json on the Spark, plus explicit
#      remote.inbox / remote.pull_dirs paths under C:/Users/<owner>/...
#
# ROLLBACK (admin PowerShell):
#   Remove-LocalUser jarvis
#   Remove-Item C:\ProgramData\ssh\jarvis_authorized_keys
#   (delete the "Match User jarvis" block at the end of C:\ProgramData\ssh\sshd_config)
#   Restart-Service sshd
#   icacls "C:\Users\<owner>\jarvis-inbox"  /remove:g jarvis   (same for the other three)
param(
    [Parameter(Mandatory = $true)][string]$Owner,
    [Parameter(Mandatory = $true)][string]$PublicKey,
    [string]$User = "jarvis",
    [switch]$RemoveAdminKey
)
$ErrorActionPreference = "Stop"
$keyFile = "C:\ProgramData\ssh\jarvis_authorized_keys"
$sshdConfig = "C:\ProgramData\ssh\sshd_config"
$adminKeys = "C:\ProgramData\ssh\administrators_authorized_keys"
$home_ = "C:\Users\$Owner"

function Say($s) { Write-Output ("[jarvis-user] " + $s) }

if ($RemoveAdminKey) {
    # Step 3: drop ONLY the line holding this public key, from the admin file
    # and from the owner's own authorized_keys if it is there too.
    $keyBody = ($PublicKey -split "\s+")[1]
    foreach ($f in @($adminKeys, "$home_\.ssh\authorized_keys")) {
        if (Test-Path $f) {
            $lines = @(Get-Content $f)
            $kept = @($lines | Where-Object { $_ -notmatch [regex]::Escape($keyBody) })
            Set-Content -Path $f -Value $kept -Encoding ascii
            Say ("removed " + ($lines.Count - $kept.Count) + " line(s) from " + $f + "; " + $kept.Count + " remain")
        }
    }
    exit 0
}

# Step 1a: the account. Users group only. Key-only in practice: a random
# 32-char password nobody keeps, and sshd refuses password auth for it below.
$existing = Get-LocalUser -Name $User -ErrorAction SilentlyContinue
if (-not $existing) {
    $bytes = New-Object byte[] 24
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $pw = [Convert]::ToBase64String($bytes)
    New-LocalUser -Name $User -Password (ConvertTo-SecureString $pw -AsPlainText -Force) `
        -PasswordNeverExpires -UserMayNotChangePassword -AccountNeverExpires `
        -Description "Jarvis file link (key-only, standard user)" | Out-Null
    Remove-Variable pw, bytes
    Say "created local user $User"
} else {
    Say "user $User already exists; leaving it"
}
if (-not (Get-LocalGroupMember -Group Users -Member $User -ErrorAction SilentlyContinue)) {
    Add-LocalGroupMember -Group Users -Member $User
}
if (Get-LocalGroupMember -Group Administrators -Member $User -ErrorAction SilentlyContinue) {
    Remove-LocalGroupMember -Group Administrators -Member $User
    Say "REMOVED $User from Administrators (it must never be one)"
}
Say ("is_admin=" + [bool](Get-LocalGroupMember -Group Administrators -Member $User -ErrorAction SilentlyContinue))

# Step 1b: the key file, readable by SYSTEM and Administrators only (sshd
# refuses a key file others can write).
Set-Content -Path $keyFile -Value $PublicKey -Encoding ascii
icacls $keyFile /inheritance:r /grant "SYSTEM:F" /grant "Administrators:F" | Out-Null
Say "wrote $keyFile (1 key)"

# Step 1c: the Match block, appended at the END (a Match block runs to the
# end of the file, so it can only go last).
$cfg = Get-Content $sshdConfig -Raw
if ($cfg -notmatch "(?m)^\s*Match User $User\b") {
    $block = "`r`nMatch User $User`r`n    AuthorizedKeysFile __PROGRAMDATA__/ssh/jarvis_authorized_keys`r`n    PasswordAuthentication no`r`n"
    Add-Content -Path $sshdConfig -Value $block -Encoding ascii
    Say "appended Match User $User block to sshd_config"
} else {
    Say "sshd_config already has a Match User $User block"
}
# Validate before restarting: a broken config would lock every ssh login out.
$sshd = Join-Path $env:ProgramFiles "OpenSSH\sshd.exe"
if (-not (Test-Path $sshd)) { $sshd = Join-Path $env:windir "System32\OpenSSH\sshd.exe" }
& $sshd -t -f $sshdConfig
if ($LASTEXITCODE -ne 0) { throw "sshd_config failed validation; NOT restarting sshd" }
Say "sshd_config validates"

# Step 1d: folders and rights. Traverse is implicit (Users have Bypass
# traverse checking), so only the four folders themselves are granted.
foreach ($f in @("$home_\jarvis-inbox", "$home_\jarvis-outbox")) {
    if (-not (Test-Path $f)) { New-Item -ItemType Directory -Path $f | Out-Null; Say "created $f" }
}
icacls "$home_\jarvis-inbox"  /grant "${User}:(OI)(CI)M" | Out-Null
icacls "$home_\jarvis-outbox" /grant "${User}:(OI)(CI)RX" | Out-Null
icacls "$home_\Desktop"       /grant "${User}:(OI)(CI)RX" | Out-Null
icacls "$home_\Downloads"     /grant "${User}:(OI)(CI)RX" | Out-Null
Say "granted: inbox=modify outbox=read desktop=read downloads=read"

# Step 1e: apply. The admin session that ran this may drop; the Spark
# re-checks from its side.
Restart-Service sshd
Say "sshd restarted; verify from the Spark: ssh -i ~/.ssh/hpcomputer $User@<host> whoami"

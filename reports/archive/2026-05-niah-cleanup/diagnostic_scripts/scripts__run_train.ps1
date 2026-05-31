$logFile = "e:\Project\python\DSRA\scripts\train_output.log"
$python = "e:\Project\python\DSRA\.env\scripts\python.exe"
$script = "e:\Project\python\DSRA\scripts\quick_train_test.py"

"Starting training at $(Get-Date)" | Out-File $logFile
& $python -u $script 2>&1 | Out-File $logFile -Append
"Exit code: $LASTEXITCODE" | Out-File $logFile -Append
"Finished at $(Get-Date)" | Out-File $logFile -Append

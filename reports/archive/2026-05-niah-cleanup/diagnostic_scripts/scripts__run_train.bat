@echo off
cd /d e:\Project\python\DSRA
echo Starting training at %DATE% %TIME% > scripts\train_output.log
"e:\Project\python\DSRA\.env\scripts\python.exe" -u scripts\quick_train_test.py >> scripts\train_output.log 2>&1
echo Exit code: %ERRORLEVEL% >> scripts\train_output.log
echo Finished at %DATE% %TIME% >> scripts\train_output.log

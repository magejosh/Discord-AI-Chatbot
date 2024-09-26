@echo off

:: Activate the virtual environment
call venv\Scripts\activate

:: Run the bot
python main.py

:: Check for errors
IF %ERRORLEVEL% NEQ 0 (
    echo There was an error running the bot.
    echo Exiting...
    exit /b %ERRORLEVEL%
)

:: Deactivate the virtual environment
deactivate

:: Keep the console open until user input
pause

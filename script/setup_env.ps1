# Load environment variables from .env file
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim()
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
        Write-Host "Set $name" -ForegroundColor Green
    }
}

Write-Host "`nEnvironment variables loaded successfully!" -ForegroundColor Cyan
Write-Host "You can now run:" -ForegroundColor Yellow
Write-Host "  - python -m extract.cli setup" -ForegroundColor White
Write-Host "  - python -m extract.cli crawl" -ForegroundColor White
Write-Host "  - cd dbt_project; dbt debug" -ForegroundColor White

$envFile = ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)\s*=\s*(.+)\s*$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            # Set for both process and session
            [System.Environment]::SetEnvironmentVariable($name, $value, [System.EnvironmentVariableTarget]::Process)
            Set-Item -Path "env:$name" -Value $value -ErrorAction SilentlyContinue
            Write-Host "Loaded: $name" -ForegroundColor Green
        }
    }
    Write-Host ""
}

Set-Location dbt_project

dbt debug

dbt deps

dbt test --select source:*

dbt run --select staging.*

dbt run --select intermediate.*

dbt run --select mart.*

dbt test

dbt docs generate

Set-Location ..

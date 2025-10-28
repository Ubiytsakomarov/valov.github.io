param(
    [string]$Owner = "valov",
    [string]$Repository = "valov.github.io",
    [string]$Branch = "main",
    [string]$Output = "flux-workflow.json"
)

$uri = "https://raw.githubusercontent.com/$Owner/$Repository/$Branch/flux-workflow.json"

Write-Host "Downloading $uri ..."
try {
    Invoke-WebRequest -Uri $uri -OutFile $Output -UseBasicParsing
    Write-Host "Saved to" (Resolve-Path $Output)
} catch {
    Write-Error "Failed to download workflow JSON. $_"
    exit 1
}

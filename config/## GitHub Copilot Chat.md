## GitHub Copilot Chat

- Extension Version: 0.28.5 (prod)
- VS Code: vscode/1.101.2
- OS: Linux

## Network

User Settings:
```json
  "github.copilot.advanced.debug.useElectronFetcher": true,
  "github.copilot.advanced.debug.useNodeFetcher": false,
  "github.copilot.advanced.debug.useNodeFetchFetcher": true
```

Connecting to https://api.github.com:
- DNS ipv4 Lookup: 188.40.181.52 (17 ms)
- DNS ipv6 Lookup: Error (37 ms): getaddrinfo ENOTFOUND api.github.com
- Proxy URL: None (1 ms)
- Electron fetch (configured): HTTP 403 (366 ms)
- Node.js https: HTTP 403 (440 ms)
- Node.js fetch: HTTP 403 (414 ms)
- Helix fetch: HTTP 403 (619 ms)

Connecting to https://api.individual.githubcopilot.com/_ping:
- DNS ipv4 Lookup: 142.54.189.108 (15 ms)
- DNS ipv6 Lookup: Error (70 ms): getaddrinfo ENOTFOUND api.individual.githubcopilot.com
- Proxy URL: None (25 ms)
- Electron fetch (configured): HTTP 200 (1182 ms)
- Node.js https: HTTP 200 (3055 ms)
- Node.js fetch: HTTP 200 (7131 ms)
- Helix fetch: timed out after 10 seconds

## Documentation

In corporate networks: [Troubleshooting firewall settings for GitHub Copilot](https://docs.github.com/en/copilot/troubleshooting-github-copilot/troubleshooting-firewall-settings-for-github-copilot).
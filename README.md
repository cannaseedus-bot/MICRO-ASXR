# MICRO.ASXR — Distributed Self-Improving Operating System
_A cognitive OS capsule that runs in the browser, learns from experience, self-corrects, and syncs its mind across devices._

## Run Locally (no Python needed)
```bash
npx serve .
# or
npx http-server .
```
Open the printed URL (e.g., http://localhost:3000).

## Key Routes (inside the OS)
- `/` — Home
- `/about` — About
- `/telemetry` — System status
- `/terminal` — Basher (trainer shell)
- `/lan` — Multi-node sync console
- `/tutorial` — Blueprint guided onboarding

## Optional JSON Relay (ASX-SERVER style)
```bash
cd server
npx json-server --watch db.json --port 8800
```
Then set `remote_api` in the capsule’s `/db/lan_settings.json`:
```json
{ "announce": true, "sync_brain": true, "sync_notes": true, "remote_api": "http://<LAN-IP>:8800" }
```

## Create Agents (in Basher)
```
agent define navigator RouteMaster
agent define archivist MemoryIndex
agent define builder UIConstructor
agent define observer TelemetryMonitor
teach builder constructs UI blocks
teach navigator resolves uncertain navigation
```

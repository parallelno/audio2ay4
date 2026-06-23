# chat-history — moving Copilot context to another machine

This folder travels with the git repo so the assistant's context can be restored on a second
machine (the GPU box used for training).

## Contents
- `ba34993c-9bcb-4eaf-8d0c-7c7ee3af9332.jsonl` — the full Copilot Chat **session** for this project
  (the conversation that designed/implemented Plan A warm-start). Re-importable into VS Code.
- `audio2ay4-memory.md` — snapshot of the assistant's **repository memory** (verified project
  facts, the running design/decision log). Paste this back into repo memory on the new machine.

## Restore the chat session on the new machine
The chat lives under the workspace-storage folder, keyed by a hash of the workspace path.

1. On the new machine, open the project folder (`audio2ay4`) in VS Code once, then close VS Code
   (this creates the workspace-storage entry).
2. Find that machine's storage hash for this workspace:
   ```powershell
   Get-ChildItem "$env:APPDATA\Code\User\workspaceStorage\*\workspace.json" |
     Select-String -Pattern "audio2ay4" | Select-Object Path
   ```
3. Copy the `.jsonl` into that folder's `chatSessions\` (create it if missing):
   ```powershell
   $hashDir = Split-Path (Get-ChildItem "$env:APPDATA\Code\User\workspaceStorage\*\workspace.json" |
     Select-String "audio2ay4").Path
   New-Item -ItemType Directory -Force "$hashDir\chatSessions" | Out-Null
   Copy-Item ".\chat-history\ba34993c-9bcb-4eaf-8d0c-7c7ee3af9332.jsonl" "$hashDir\chatSessions\"
   ```
4. Launch VS Code, open the folder, open the Chat view, and pick the conversation from history
   ("Show Chats" / the history icon).

Notes: VS Code + Copilot Chat must be the same or a newer version. If the project path differs on
the new machine, the hash differs — that's why step 2 looks it up rather than hard-coding it.

## Restore the assistant's repository memory
Repo memory is workspace-local and not part of the chat session, so import it separately: on the
new machine, tell Copilot to **"load `chat-history/audio2ay4-memory.md` into repository memory"**, or
paste its contents into `/memories/repo/audio2ay4.md` via the memory tool. The assistant keeps this
file updated as the single source of verified project facts and the running decision log.

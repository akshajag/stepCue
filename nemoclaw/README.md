# StepCue in NVIDIA NemoClaw

Run StepCue's caregiver assistant as an OpenClaw skill inside an NVIDIA OpenShell
sandbox. Health data deserves a locked room: the sandbox is deny-by-default, the only
doors we open are NVIDIA's model endpoints, the API key never enters the sandbox
(OpenShell injects it at the egress point), and the data we copy in has no names and
no location.

| File | What it is |
| --- | --- |
| `skills/stepcue-care/SKILL.md` | The OpenClaw skill: when to use it, which StepCue script answers what, the no-medical-advice rules |
| `presets/stepcue-nvidia.yaml` | OpenShell egress preset: only `integrate.api.nvidia.com` (chat, embeddings), `ai.api.nvidia.com` (rerank) and `grpc.nvcf.nvidia.com` (Riva TTS), only for `python3` |
| `make_snapshot.py` | Exports episodes, agent decisions and check-ins, scrubbed of names, emails, phone numbers and coordinates |

## Run it (Brev launchable or any Linux box with NemoClaw)

This laptop is Windows without an NVIDIA GPU, so use the hackathon's Brev launchable
(brev.dev, "NemoClaw sandbox") or a DGX Spark.

1. On the laptop, export the snapshot:
   ```
   python nemoclaw/make_snapshot.py
   ```
2. On the Brev box, onboard NemoClaw with Nemotron as the agent's brain:
   ```
   export NVIDIA_INFERENCE_API_KEY=nvapi-...      # your key, typed on the Brev box
   NEMOCLAW_PROVIDER=build NEMOCLAW_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b nemoclaw onboard --name stepcue
   ```
3. Copy the project in, add the egress preset, install the skill:
   ```
   nemoclaw stepcue policy add --from-file nemoclaw/presets/stepcue-nvidia.yaml --dry-run
   nemoclaw stepcue policy add --from-file nemoclaw/presets/stepcue-nvidia.yaml --yes
   nemoclaw stepcue policy add pypi --yes            # once, to pip install openai inside
   nemoclaw stepcue skill install nemoclaw/skills/stepcue-care/
   nemoclaw stepcue connect
   ```
   Inside the sandbox put the StepCue `.py` files, `knowledge/` and the `snapshot/` folder in
   `~/stepcue`, then `pip install openai httpx`.
4. Ask the OpenClaw agent: "How did walking go today?" or "Write today's care report".
   It runs the skill, which calls Nemotron (agent and Ultra), Nemotron Embed and Rerank, and
   the guard models through the preset, and answers with citations.

Check what the sandbox may reach: `nemoclaw stepcue policy list` and
`nemoclaw stepcue policy explain stepcue-nvidia`.

Not confirmed on real hardware yet: gRPC to NVCF through `tls: skip` (Riva TTS is optional
inside the sandbox; the voice cues are normally made on the laptop).

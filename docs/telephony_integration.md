# Telephony Integration Guide

## Overview

This document describes **where** and **how** a real telephony system would
integrate with the VoiceGuard detection pipeline. All integration points are
marked with `# INTEGRATION_POINT` comments in the source code.

---

## Architecture position

```
[PSTN / SIP trunk]
        │  RTP / SRTP audio
        ▼
[Telephony Layer — NOT BUILT]    ← This document describes this layer
        │  PCM chunks (16-bit, 16 kHz)
        ▼
[demo/server.py WebSocket /ws/stream]  ← Detection pipeline starts here
```

---

## Option A — Asterisk AGI / ARI

**Use when:** You control an Asterisk PBX and want in-path detection.

### Integration pattern

```python
# Asterisk ARI (REST Interface) — attach a media bridge to a channel
# and stream audio to the detection service via WebSocket.

import ari
import websockets
import asyncio

client = ari.connect('http://asterisk:8088/', 'user', 'pass')

def on_stasis_start(channel, ev):
    # 1. Answer the channel
    channel.answer()

    # 2. Open a WebSocket to VoiceGuard
    async def stream_to_voiceguard():
        uri = "ws://voiceguard:8000/ws/stream"
        async with websockets.connect(uri) as ws:
            # 3. Subscribe to channel audio events via ARI /channels/{id}/audio
            #    (or use an AudioSocket application in Asterisk)
            #    Forward PCM chunks to VoiceGuard, receive decisions
            async for decision in receive_decisions(ws):
                if decision["action"] == "BLOCK":
                    channel.hangup()
                elif decision["action"] == "ALERT":
                    notify_fraud_agent(channel.id, decision)

    asyncio.run(stream_to_voiceguard())

client.on_channel_event('StasisStart', on_stasis_start)
client.run(apps='voiceguard-app')
```

**Asterisk dialplan hook:**
```
; extensions.conf
exten => _X.,1,Stasis(voiceguard-app)
```

**AudioSocket alternative (lower latency):**
Use the Asterisk `AudioSocket` application to stream raw PCM directly to
a TCP endpoint, avoiding HTTP overhead:
```
exten => _X.,1,AudioSocket(voiceguard-host,9093)
```
The VoiceGuard server would then accept TCP connections and parse the
AudioSocket protocol instead of WebSocket.

---

## Option B — FreeSWITCH ESL (Event Socket Layer)

**Use when:** You use FreeSWITCH as your media server.

```python
from ESL import ESLconnection

con = ESLconnection("freeswitch-host", "8021", "ClueCon")
con.events("plain", "CUSTOM media::base64")

def on_media_event(e):
    # FreeSWITCH can be configured to emit base64-encoded audio chunks
    # via mod_event_socket + mod_audio_stream
    import base64, asyncio, websockets
    audio_b64 = e.getBody()
    pcm_bytes  = base64.b64decode(audio_b64)
    # Forward pcm_bytes → VoiceGuard WebSocket /ws/stream
    ...
```

**FreeSWITCH dialplan (mod_audio_stream):**
```xml
<extension name="voiceguard">
  <condition field="destination_number" expression="^(\d+)$">
    <action application="answer"/>
    <action application="audio_stream"
            data="ws://voiceguard:8000/ws/stream"/>
  </condition>
</extension>
```

---

## Option C — Twilio Webhook (Cloud Telephony)

**Use when:** You use Twilio Programmable Voice and want cloud-hosted detection.

```python
# FastAPI webhook — Twilio calls this on incoming call
from fastapi import Request
from fastapi.responses import Response

@app.post("/twilio/incoming")
async def twilio_incoming(request: Request):
    # 1. Return TwiML to start a media stream
    twiml = """
    <Response>
      <Start>
        <Stream url="wss://voiceguard.yourdomain.com/ws/twilio"
                track="inbound_track" />
      </Start>
      <Say>Please hold while we verify your identity.</Say>
    </Response>
    """
    return Response(content=twiml, media_type="application/xml")

# Twilio sends audio as base64-encoded mulaw (8 kHz) over WebSocket
@app.websocket("/ws/twilio")
async def ws_twilio(ws: WebSocket):
    await ws.accept()
    import base64, audioop
    while True:
        raw = await ws.receive()
        if "text" in raw:
            msg = json.loads(raw["text"])
            if msg.get("event") == "media":
                mulaw = base64.b64decode(msg["media"]["payload"])
                # Convert mulaw 8kHz → linear PCM 16kHz
                pcm8   = audioop.ulaw2lin(mulaw, 2)
                pcm16  = audioop.ratecv(pcm8, 2, 1, 8000, 16000, None)[0]
                # Feed into the same scoring pipeline
                await forward_to_scorer(pcm16)
```

**Key differences from direct WebSocket:**
- Twilio sends μ-law 8 kHz audio (must upsample to 16 kHz)
- Audio arrives as base64 JSON events, not raw binary
- Decisions must be enacted via Twilio's REST API (hangup, conference-insert)

---

## Decision → Action Mapping

| VoiceGuard Decision | Telephony action |
|---|---|
| `ALLOW` | Continue call normally |
| `STEP_UP` | Play IVR challenge prompt; capture and verify response |
| `ALERT` | Whisper alert to agent via conference bridge; flag in CRM |
| `BLOCK` | Disconnect call (`channel.hangup()` / `<Hangup/>` TwiML) |

---

## Data Flow Diagram

```
PSTN caller
    │ voice
    ▼
[Telco gateway]
    │ SIP + RTP
    ▼
[Asterisk / FreeSWITCH / Twilio]
    │ PCM chunks (16-bit / 16 kHz)   via AudioSocket / WebSocket / ESL
    ▼
[VoiceGuard /ws/stream]
    │ JSON score update every ~1s
    ▼
[PolicyEngine.decide()]
    │ Decision: ALLOW / STEP_UP / ALERT / BLOCK
    ▼
[Telephony action]   →   [CRM / fraud queue alert]
                     →   [DecisionLogger (audit trail)]
```

---

## Latency Consideration

The architecture §4 latency budget is **350–600ms per score update**. This is
acceptable for call monitoring (score updates stream throughout the call) but
means the *first* decision arrives ~1–2s after the call begins. Design the IVR
flow to fill this gap (greeting message, hold music) so the first human-to-human
exchange is already being scored by the time it matters.

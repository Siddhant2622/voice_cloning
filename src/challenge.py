"""
src/challenge.py
Layer 4c — Active challenge-response (anti-replay / anti-pre-recorded injection).

STUB: This module provides the interface described in the architecture.
In a production deployment this would integrate with:
  - A real ASR engine (Whisper, Google STT, Azure Speech) for transcript comparison
  - A TTS engine to speak the challenge over the call
  - The telephony layer (Asterisk AGI / FreeSWITCH ESL / Twilio TwiML) to inject
    the challenge prompt and capture the caller's response

Architecture reference: Layer 4c (active challenge) in voice-clone-detection-architecture.md
"""

from __future__ import annotations

import logging
import random
import string
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Challenge wordlists
# ---------------------------------------------------------------------------
# These phrases are nonsensical / not common utterances to defeat
# pre-recorded snippet injection. They should be regenerated per call.
CHALLENGE_WORDLIST = [
    "purple umbrella seven",
    "crinkle forty jasmine",
    "quartz pillow midnight",
    "frozen anchor delta",
    "silver pencil eighty",
    "crimson lantern five",
    "wobble sparrow nine",
    "tangled marble june",
    "hollow cactus three",
    "blizzard anchor fox",
]

DIGIT_POOL = string.digits   # 0-9


@dataclass
class Challenge:
    challenge_id: str
    phrase:       Optional[str]   # e.g. "purple umbrella seven"
    digits:       Optional[str]   # e.g. "7 4 2 9"
    issued_at:    float = field(default_factory=time.time)
    expires_in_s: int = 30        # challenge is valid for 30 seconds


@dataclass
class ChallengeResult:
    challenge_id:  str
    passed:        bool
    confidence:    float   # 0 = total mismatch, 1 = perfect match
    method:        str     # "phrase" | "digits"
    details:       str     # human-readable explanation
    score_delta:   float   # suggested additive spoof-score adjustment if failed


# ---------------------------------------------------------------------------
# Challenge generator
# ---------------------------------------------------------------------------
class ChallengeGenerator:
    """
    Generates unique, randomised challenge prompts.
    Each challenge has a short TTL and a unique ID.
    """

    def generate_phrase(self) -> Challenge:
        """Generate a random-phrase challenge."""
        return Challenge(
            challenge_id=str(uuid.uuid4()),
            phrase=random.choice(CHALLENGE_WORDLIST),
            digits=None,
        )

    def generate_digits(self, length: int = 4) -> Challenge:
        """Generate a random digit string challenge (OTP-style)."""
        digits = " ".join(random.choices(DIGIT_POOL, k=length))
        return Challenge(
            challenge_id=str(uuid.uuid4()),
            phrase=None,
            digits=digits,
        )

    def generate(self, mode: str = "digits") -> Challenge:
        """
        Generate a challenge prompt.

        Args:
            mode: "digits" (default, harder for TTS replay) or "phrase"

        Returns:
            Challenge object with id, content, and expiry.
        """
        if mode == "phrase":
            return self.generate_phrase()
        return self.generate_digits()


# ---------------------------------------------------------------------------
# Challenge verifier
# ---------------------------------------------------------------------------
class ChallengeVerifier:
    """
    Verifies a caller's response to a challenge.

    STUB NOTE: In production, `transcribed_response` would come from a real ASR
    engine. The comparison here is case-insensitive exact / fuzzy string match.
    Replace the _asr_transcribe() stub with a real ASR call.

    INTEGRATION_POINT:
        1. Telephony layer captures audio after challenge is played.
        2. Audio is sent to ASR (e.g. `openai/whisper-base` or cloud STT).
        3. Transcript is passed to ChallengeVerifier.verify().
        4. Result affects the final risk score.
    """

    def verify(
        self,
        challenge: Challenge,
        transcribed_response: Optional[str] = None,
        audio_waveform: Optional[object] = None,  # torch.Tensor at 16 kHz
    ) -> ChallengeResult:
        """
        Verify a challenge response.

        Args:
            challenge:            The issued Challenge.
            transcribed_response: ASR transcript of caller's response (STUB: manually set).
            audio_waveform:       Raw audio for ASR (STUB: not processed here).

        Returns:
            ChallengeResult with pass/fail and confidence.
        """
        # Check expiry
        if time.time() - challenge.issued_at > challenge.expires_in_s:
            return ChallengeResult(
                challenge_id=challenge.challenge_id,
                passed=False,
                confidence=0.0,
                method="digits" if challenge.digits else "phrase",
                details="Challenge expired before a response was received.",
                score_delta=0.10,   # mild penalty for no-response
            )

        expected = (challenge.digits or challenge.phrase or "").strip().lower()

        if transcribed_response is None:
            # STUB: no ASR result provided — this is the demo path
            logger.warning(
                "ChallengeVerifier: no ASR transcript provided. "
                "STUB: treating as a neutral (no-result) outcome. "
                "INTEGRATION_POINT: wire in real ASR engine here."
            )
            return ChallengeResult(
                challenge_id=challenge.challenge_id,
                passed=False,
                confidence=0.0,
                method="digits" if challenge.digits else "phrase",
                details=(
                    "STUB — no ASR transcript available. "
                    "In production: transcribe caller audio and compare."
                ),
                score_delta=0.0,
            )

        response = transcribed_response.strip().lower()

        # Exact match
        if response == expected:
            return ChallengeResult(
                challenge_id=challenge.challenge_id,
                passed=True,
                confidence=1.0,
                method="digits" if challenge.digits else "phrase",
                details=f"Exact match: '{response}'",
                score_delta=-0.05,  # slight score reduction (genuine indicator)
            )

        # Fuzzy match (edit-distance ratio)
        ratio = _edit_similarity(expected, response)
        if ratio >= 0.85:
            return ChallengeResult(
                challenge_id=challenge.challenge_id,
                passed=True,
                confidence=ratio,
                method="digits" if challenge.digits else "phrase",
                details=f"Fuzzy match ({ratio:.0%}): '{response}' vs '{expected}'",
                score_delta=-0.02,
            )
        elif ratio >= 0.50:
            return ChallengeResult(
                challenge_id=challenge.challenge_id,
                passed=False,
                confidence=ratio,
                method="digits" if challenge.digits else "phrase",
                details=f"Partial match ({ratio:.0%}): '{response}' vs '{expected}' — insufficient.",
                score_delta=0.15,
            )
        else:
            return ChallengeResult(
                challenge_id=challenge.challenge_id,
                passed=False,
                confidence=ratio,
                method="digits" if challenge.digits else "phrase",
                details=f"Mismatch ({ratio:.0%}): '{response}' vs '{expected}'.",
                score_delta=0.25,
            )


def _edit_similarity(a: str, b: str) -> float:
    """Normalized Levenshtein similarity."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return 1.0 - dp[n] / max(m, n)

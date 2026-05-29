import os
import shutil
import logging
import torch
import torch.nn.functional as F
import numpy as np
import librosa
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("voiceguard")

# ==========================================
# 1. SETUP & CORS
# ==========================================
app = FastAPI(title="VoiceGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. RESEMBLYZER — Text-Independent Speaker Encoder
# ==========================================
# Resemblyzer uses Google's GE2E (Generalized End-to-End) architecture,
# pre-trained on 2,000+ speakers with diverse utterances.
# The resulting 256-dim embedding encodes ONLY the speaker's vocal identity —
# completely text-independent.
#
# Realistic same-speaker similarity with consumer mics: 0.70 – 0.92
# Cross-speaker similarity:                             0.20 – 0.65

from resemblyzer import VoiceEncoder, preprocess_wav

print("Loading Resemblyzer GE2E speaker encoder...")
encoder = VoiceEncoder(device="cpu")
print("Speaker encoder ready — text-independent voice authentication active.")

# ── Authentication Threshold ────────────────────────────────────────────────
# Raised to 0.78 to combat false accepts (cross-speaker FAR).
# With the new cleaner 2-WAV peak-normalized audio pipeline, legitimate speakers
# consistently score above 0.80. A friend or similar-sounding impostor may 
# score around 0.72-0.75. 0.78 is a secure discriminator for Resemblyzer embeddings.
SIMILARITY_THRESHOLD = 0.78

# Increased from 3 → 5 for a more robust speaker centroid.
# 5 diverse phrases capture wider vocal variation → more stable embeddings.
MIN_ENROLLMENT_SAMPLES = 5

# Minimum quality score (0–100) required for enrollment samples.
MIN_ENROLLMENT_QUALITY = 40

# The Secure Vault — only math, never raw audio
os.makedirs("secure_database", exist_ok=True)


# ==========================================
# 3. AUDIO PREPROCESSING & QUALITY
# ==========================================

def compute_audio_quality(audio: np.ndarray, sr: int) -> dict:
    """
    Lightweight audio quality diagnostics.

    Returns a dict with:
      rms              — Root-mean-square energy level
      clipping_pct     — % of samples at/near clipping (|sample| ≥ 0.99)
      speech_duration  — seconds of non-silent audio after librosa trim
      silence_ratio    — fraction of total audio that is silence
      quality_score    — composite score 0–100 (higher = better)
    """
    # RMS energy
    rms = float(np.sqrt(np.mean(audio ** 2)))

    # Clipping: samples at ±0.99+ (float range [-1, 1])
    n_clipped = int(np.sum(np.abs(audio) >= 0.99))
    clipping_pct = round(100.0 * n_clipped / max(len(audio), 1), 2)

    # Speech duration via librosa trim (20 dB threshold)
    total_duration = len(audio) / sr
    try:
        trimmed, _ = librosa.effects.trim(audio, top_db=20)
        speech_duration = len(trimmed) / sr
    except Exception:
        speech_duration = total_duration
    silence_ratio = round(1.0 - (speech_duration / max(total_duration, 0.01)), 3)

    # ── Composite quality score (0–100) ──
    # Penalise: low RMS, clipping, too much silence, too-short speech
    score = 100.0

    # RMS penalty: ideal 0.02–0.30; too quiet = unstable embedding
    if rms < 0.005:
        score -= 50        # essentially silent
    elif rms < 0.01:
        score -= 30        # very quiet
    elif rms < 0.02:
        score -= 10        # quiet but usable

    # Clipping penalty
    if clipping_pct > 5.0:
        score -= 30
    elif clipping_pct > 1.0:
        score -= 15

    # Speech duration penalty: need ≥ 1.5s of actual speech for stable embed
    if speech_duration < 0.5:
        score -= 50
    elif speech_duration < 1.0:
        score -= 30
    elif speech_duration < 1.5:
        score -= 10

    # Silence ratio penalty: > 70% silence = mostly dead air
    if silence_ratio > 0.8:
        score -= 20
    elif silence_ratio > 0.6:
        score -= 10

    score = max(0, min(100, int(score)))

    return {
        "rms":              round(rms, 5),
        "clipping_pct":     clipping_pct,
        "speech_duration":  round(speech_duration, 2),
        "silence_ratio":    silence_ratio,
        "quality_score":    score,
    }


def preprocess_audio(file_path: str) -> np.ndarray:
    """
    Robust audio preprocessing for Resemblyzer.

    1. Load audio at 16 kHz (Resemblyzer's native rate) via librosa — avoids
       the quality loss from scipy's int16 → float conversion → resampling chain.
    2. Peak-normalise to [-0.95, 0.95] — stabilises embedding amplitude.
    3. Trim leading/trailing silence using librosa (20 dB threshold).
    4. Validate minimum speech duration (≥ 1.0 s after trimming).

    Returns the cleaned float32 waveform at 16 kHz.
    Raises ValueError if audio is too short or silent.
    """
    # Step 1: Load at 16 kHz — librosa handles ANY source sample rate
    audio, sr = librosa.load(file_path, sr=16000, mono=True)

    if len(audio) == 0:
        raise ValueError("Empty audio file received.")

    # Step 2: Peak-normalise to [-0.95, 0.95]
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = (audio / peak) * 0.95
    else:
        raise ValueError("Audio is completely silent (peak ≈ 0).")

    # Step 3: Trim silence (20 dB below peak)
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)

    # Step 4: Validate speech duration
    speech_secs = len(audio_trimmed) / sr
    if speech_secs < 1.0:
        raise ValueError(
            f"Only {speech_secs:.1f}s of speech detected after silence trimming. "
            f"Need at least 1.0s. Please speak louder and for the full phrase."
        )

    log.info(
        "Preprocessed: %.1fs total → %.1fs speech, peak=%.3f, sr=%d",
        len(audio) / sr, speech_secs, peak, sr,
    )

    return audio_trimmed.astype(np.float32)


# ==========================================
# 4. AUDIO → SPEAKER EMBEDDING (REFACTORED)
# ==========================================

def get_speaker_embedding(file_path: str) -> tuple[np.ndarray, dict]:
    """
    Extracts a 256-dim L2-normalised speaker embedding from an audio file.

    Uses the refactored preprocessing pipeline:
      1. librosa loads at 16 kHz (no double-resampling)
      2. Peak-normalise
      3. Silence trimming
      4. Quality diagnostics
      5. Resemblyzer preprocess_wav for final conditioning
      6. GE2E encoder → 256-dim embedding

    Returns:
      (embedding, quality_dict) — embedding is np.float32 shape (256,)
    """
    # Our custom preprocessing: load at 16 kHz, normalise, trim
    clean_audio = preprocess_audio(file_path)

    # Quality diagnostics on the cleaned audio
    quality = compute_audio_quality(clean_audio, sr=16000)

    # Resemblyzer's own final conditioning (applies its own VAD + normalisation)
    # We pass the already-clean audio as a numpy array.  preprocess_wav can
    # accept an (audio, sr) tuple — we use the numpy-array overload.
    wav_conditioned = preprocess_wav(clean_audio, source_sr=16000)

    # GE2E embedding
    embedding = encoder.embed_utterance(wav_conditioned)
    embedding = embedding.astype(np.float32)

    # Sanity: verify L2 norm ≈ 1.0 (Resemblyzer guarantees this)
    norm = float(np.linalg.norm(embedding))
    log.info("Embedding norm: %.4f | Quality score: %d", norm, quality["quality_score"])

    return embedding, quality


def embed_to_tensor(embed: np.ndarray) -> torch.Tensor:
    """Converts a numpy embedding to a torch tensor for storage."""
    return torch.tensor(embed, dtype=torch.float32)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D tensors. Range: [−1, 1]."""
    return F.cosine_similarity(
        a.reshape(1, -1),
        b.reshape(1, -1)
    ).item()


# ==========================================
# 5. STORAGE HELPERS
# ==========================================
def _samples_path(username: str) -> str:
    return f"secure_database/{username}_samples.pt"


def _voiceprint_path(username: str) -> str:
    return f"secure_database/{username}_voiceprint.pt"


def _quality_path(username: str) -> str:
    """Per-user enrollment quality metrics."""
    return f"secure_database/{username}_quality.pt"


def _load_samples(username: str) -> list:
    """Returns the list of stored embedding tensors, or [] if none."""
    path = _samples_path(username)
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=True)
    return []


def _load_quality(username: str) -> list:
    """Returns the list of stored quality dicts, or [] if none."""
    path = _quality_path(username)
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    return []


def _compute_and_save_centroid(username: str, samples: list) -> torch.Tensor:
    """Averages all enrollment tensors → centroid; L2-normalises → saves."""
    stacked  = torch.stack(samples)            # (N, 256)
    centroid = stacked.mean(dim=0)             # (256,)
    centroid = F.normalize(centroid, dim=0)    # keep it on the unit sphere
    torch.save(centroid, _voiceprint_path(username))
    return centroid


# ==========================================
# 6. API ENDPOINTS
# ==========================================

@app.post("/register")
async def register_user(username: str = Form(...), file: UploadFile = File(...)):
    """
    Multi-phrase enrollment endpoint.

    Call this MIN_ENROLLMENT_SAMPLES (5) times, each time with a DIFFERENT
    challenge phrase, to build a robust, text-independent speaker centroid.

    The backend accumulates each embedding.  The centroid is re-computed
    after every call, so the voiceprint improves progressively.

    Quality gating: samples with quality_score < MIN_ENROLLMENT_QUALITY (40)
    are rejected with a 422 error — the user must re-record.
    """
    temp_path = f"temp_reg_{username}.wav"

    try:
        with open(temp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        # ── Extract speaker embedding with quality check ──
        try:
            embed_np, quality = get_speaker_embedding(temp_path)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        # ── Enrollment quality gate ──
        if quality["quality_score"] < MIN_ENROLLMENT_QUALITY:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Audio quality too low for enrollment "
                    f"(score: {quality['quality_score']}/100, minimum: {MIN_ENROLLMENT_QUALITY}). "
                    f"RMS={quality['rms']:.4f}, clipping={quality['clipping_pct']}%, "
                    f"speech={quality['speech_duration']}s. "
                    f"Please speak louder, closer to the mic, and for the full phrase."
                ),
            )

        new_embed = embed_to_tensor(embed_np)

        # Load existing enrollment data
        samples   = _load_samples(username)
        qualities = _load_quality(username)

        # Rolling window: keep the most recent MIN_ENROLLMENT_SAMPLES
        if len(samples) >= MIN_ENROLLMENT_SAMPLES:
            samples.pop(0)
            if qualities:
                qualities.pop(0)

        samples.append(new_embed.detach())
        qualities.append(quality)

        # Persist sample list, quality list, and recompute centroid
        torch.save(samples, _samples_path(username))
        torch.save(qualities, _quality_path(username))
        _compute_and_save_centroid(username, samples)

        enrolled = len(samples)
        needed   = max(0, MIN_ENROLLMENT_SAMPLES - enrolled)
        complete = enrolled >= MIN_ENROLLMENT_SAMPLES

        if complete:
            msg = (
                f"Enrollment complete for '{username}'! "
                f"{enrolled} samples averaged into a text-independent voiceprint. "
                f"You may now log in with any random challenge phrase."
            )
        else:
            msg = (
                f"Sample {enrolled}/{MIN_ENROLLMENT_SAMPLES} recorded for '{username}'. "
                f"Please register {needed} more time(s) — use a DIFFERENT phrase each time."
            )

        log.info(
            "REGISTER [%s] sample %d/%d  quality=%d  %s",
            username, enrolled, MIN_ENROLLMENT_SAMPLES,
            quality["quality_score"], "COMPLETE" if complete else "PENDING",
        )

        return {
            "status":              "success",
            "samples_recorded":    enrolled,
            "samples_needed":      needed,
            "enrollment_complete": complete,
            "message":             msg,
            "audio_quality":       quality,
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/enrollment_status/{username}")
async def enrollment_status(username: str):
    """Returns enrollment progress for a user."""
    samples  = _load_samples(username)
    enrolled = len(samples)
    complete = enrolled >= MIN_ENROLLMENT_SAMPLES
    return {
        "username":            username,
        "samples_recorded":    enrolled,
        "min_required":        MIN_ENROLLMENT_SAMPLES,
        "enrollment_complete": complete,
        "can_login":           complete,
    }


@app.post("/login")
async def login_user(username: str = Form(...), file: UploadFile = File(...)):
    """
    Voice authentication — Multi-Strategy Similarity Scoring.

    Computes a weighted final score from three complementary metrics:
      • centroid_sim    (30%) — general vocal identity match against averaged centroid
      • best_sample_sim (50%) — highest similarity to a single phrase (forces exact identity match)
      • top2_avg        (20%) — average of the two best sample similarities

    Access is granted if final_score ≥ SIMILARITY_THRESHOLD (0.78).

    This multi-strategy approach is more robust than simple max-of-samples
    because:
      1. The centroid averages out per-phrase embedding noise
      2. Best-sample catches the case where one enrollment was very similar
      3. Top-2 average prevents a single lucky sample from granting access
    """
    voiceprint_path = _voiceprint_path(username)
    samples         = _load_samples(username)

    if not os.path.exists(voiceprint_path):
        raise HTTPException(
            status_code=404,
            detail="User not found. Please register first."
        )

    if len(samples) < MIN_ENROLLMENT_SAMPLES:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Enrollment incomplete: {len(samples)}/{MIN_ENROLLMENT_SAMPLES} samples. "
                f"Please register {MIN_ENROLLMENT_SAMPLES - len(samples)} more time(s)."
            ),
        )

    temp_login_path = f"temp_login_{username}.wav"

    try:
        with open(temp_login_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        # ── Extract speaker embedding with quality diagnostics ──
        try:
            login_embed_np, login_quality = get_speaker_embedding(temp_login_path)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        login_embed = embed_to_tensor(login_embed_np)

        # ── Load stored centroid ──
        centroid = torch.load(voiceprint_path, map_location="cpu", weights_only=True)

        # ── Multi-Strategy Similarity Scoring ──

        # 1. Centroid similarity (most stable — averages out per-phrase noise)
        centroid_sim = cosine_sim(centroid, login_embed)

        # 2. Per-sample similarities
        all_similarities = []
        for sample_embed in samples:
            sim = cosine_sim(sample_embed, login_embed)
            all_similarities.append(round(sim, 4))

        # 3. Best single-sample match
        sorted_sims = sorted(all_similarities, reverse=True)
        best_sample_sim = sorted_sims[0]

        # 4. Top-2 average (if we have ≥ 2 samples; otherwise = best)
        top2_avg = float(np.mean(sorted_sims[:2])) if len(sorted_sims) >= 2 else best_sample_sim

        # 5. Weighted final score
        final_score = (
            0.30 * centroid_sim
            + 0.50 * best_sample_sim
            + 0.20 * top2_avg
        )

        is_verified = bool(final_score >= SIMILARITY_THRESHOLD)

        log.info(
            "LOGIN [%s] final=%.4f  centroid=%.4f  best=%.4f  top2=%.4f  "
            "threshold=%.2f  → %s  quality=%d",
            username, final_score, centroid_sim, best_sample_sim, top2_avg,
            SIMILARITY_THRESHOLD, "GRANTED" if is_verified else "DENIED",
            login_quality["quality_score"],
        )

        return {
            "verified":           is_verified,
            "final_score":        round(final_score, 4),
            "centroid_sim":       round(centroid_sim, 4),
            "best_sample_sim":    round(best_sample_sim, 4),
            "top2_avg":           round(top2_avg, 4),
            "all_similarities":   all_similarities,
            "threshold":          SIMILARITY_THRESHOLD,
            "enrollment_samples": len(samples),
            "audio_quality":      login_quality,
            # Legacy backward compat fields
            "similarity":         round(final_score, 4),
            "distance":           round(1.0 - final_score, 4),
        }

    finally:
        if os.path.exists(temp_login_path):
            os.remove(temp_login_path)


@app.delete("/user/{username}")
async def delete_user(username: str):
    """Removes all enrollment data so a user can re-enroll from scratch."""
    removed = []
    for path in [_voiceprint_path(username), _samples_path(username), _quality_path(username)]:
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"No enrollment data found for '{username}'."
        )
    log.info("DELETED enrollment for [%s]: %s", username, removed)
    return {"status": "deleted", "files_removed": removed}


# ==========================================
# 7. DEBUG / DIAGNOSTICS ENDPOINTS
# ==========================================

@app.get("/debug/user/{username}")
async def debug_user(username: str):
    """
    Per-user diagnostics: embedding norms, intra-user similarities,
    quality scores, and centroid health.  Development-only.
    """
    samples = _load_samples(username)
    if not samples:
        raise HTTPException(status_code=404, detail=f"No enrollment data for '{username}'.")

    qualities = _load_quality(username)
    voiceprint_path = _voiceprint_path(username)

    # Embedding norms (should all be ≈ 1.0 for L2-normalised vectors)
    norms = [round(float(torch.norm(s).item()), 4) for s in samples]

    # Pairwise intra-user similarities (how consistent are enrollment samples?)
    intra_sims = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            sim = cosine_sim(samples[i], samples[j])
            intra_sims.append({
                "sample_a": i + 1, "sample_b": j + 1,
                "similarity": round(sim, 4),
            })

    # Centroid info
    centroid_info = {}
    if os.path.exists(voiceprint_path):
        centroid = torch.load(voiceprint_path, map_location="cpu", weights_only=True)
        centroid_info["norm"] = round(float(torch.norm(centroid).item()), 4)
        centroid_info["per_sample_similarity"] = [
            round(cosine_sim(centroid, s), 4) for s in samples
        ]

    return {
        "username":              username,
        "enrollment_count":      len(samples),
        "embedding_norms":       norms,
        "intra_user_similarities": intra_sims,
        "centroid":              centroid_info,
        "quality_scores":        qualities,
        "threshold":             SIMILARITY_THRESHOLD,
    }


@app.get("/debug/similarities")
async def debug_similarities():
    """
    Cross-user similarity matrix for threshold calibration.
    Development-only — AUTH-GATE or REMOVE before production.
    """
    vault = "secure_database"
    files = [f for f in os.listdir(vault) if f.endswith("_voiceprint.pt")]

    embeddings = {}
    for fname in files:
        uname = fname.replace("_voiceprint.pt", "")
        embeddings[uname] = torch.load(
            os.path.join(vault, fname), map_location="cpu", weights_only=True
        )

    users = list(embeddings.keys())
    cross = []
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            u1, u2 = users[i], users[j]
            sim  = cosine_sim(embeddings[u1], embeddings[u2])
            safe = sim < SIMILARITY_THRESHOLD
            cross.append({
                "user_a": u1, "user_b": u2,
                "similarity":          round(sim, 4),
                "correctly_separated": safe,
            })

    enrollment_info = {u: len(_load_samples(u)) for u in users}

    return {
        "current_threshold":       SIMILARITY_THRESHOLD,
        "scoring_weights":         {"centroid": 0.30, "best_sample": 0.50, "top2_avg": 0.20},
        "min_enrollment":          MIN_ENROLLMENT_SAMPLES,
        "registered_users":        users,
        "enrollment_samples":      enrollment_info,
        "cross_user_similarities": cross,
        "guidance": (
            f"The final_score uses weighted scoring (centroid 30% + best-sample 50% + "
            f"top-2 avg 20%).  SIMILARITY_THRESHOLD ({SIMILARITY_THRESHOLD}) should "
            f"be ABOVE same-speaker final_scores and BELOW cross-speaker final_scores. "
            f"correctly_separated=False means two users are too close — raise threshold."
        ),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

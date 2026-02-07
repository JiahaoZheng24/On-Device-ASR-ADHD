"""
Simple speaker diarization using acoustic features and clustering.

This is a lightweight, completely local approach that doesn't require
any external models or authentication. It uses:
1. MFCC + spectral features for speaker representation
2. K-means clustering to group speakers
3. Pitch analysis (pyin) to distinguish child vs adult voices
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import librosa

logger = logging.getLogger(__name__)


class SimpleDiarization:
    """
    Simple speaker diarization using acoustic features and clustering.

    This approach is completely local and doesn't require external models.
    """

    def __init__(
        self,
        n_speakers: int = 2,
        n_mfcc: int = 20,
        child_pitch_threshold: float = 250.0  # Hz, used as fallback
    ):
        self.n_speakers = n_speakers
        self.n_mfcc = n_mfcc
        self.child_pitch_threshold = child_pitch_threshold
        self.scaler = StandardScaler()
        self.kmeans = None

        logger.info(f"Initialized SimpleDiarization with {n_speakers} speakers")

    def extract_features(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> Tuple[np.ndarray, float]:
        """
        Extract rich acoustic features from audio segment.

        Returns:
            Tuple of (feature_vector, median_pitch)
        """
        try:
            # 1. MFCC features (speaker timbre)
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=self.n_mfcc
            )
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)

            # 2. Pitch (F0) using pyin - much more accurate than piptrack
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=60,    # Adult male lower bound
                fmax=500,   # Child upper bound
                sr=sample_rate
            )

            # Get valid (voiced) pitch values
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                median_pitch = float(np.median(valid_f0))
                pitch_mean = float(np.mean(valid_f0))
                pitch_std = float(np.std(valid_f0))
                voiced_ratio = float(np.sum(voiced_flag) / len(voiced_flag))
            else:
                median_pitch = 0.0
                pitch_mean = 0.0
                pitch_std = 0.0
                voiced_ratio = 0.0

            # 3. Spectral centroid (brightness of voice)
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio, sr=sample_rate
            )
            centroid_mean = float(np.mean(spectral_centroid))

            # 4. Zero crossing rate (noisiness / fricatives)
            zcr = librosa.feature.zero_crossing_rate(audio)
            zcr_mean = float(np.mean(zcr))

            # 5. RMS energy
            rms = librosa.feature.rms(y=audio)
            rms_mean = float(np.mean(rms))

            # Build feature vector: MFCC mean + pitch stats + spectral features
            feature_vector = np.concatenate([
                mfcc_mean,              # 20 dims: speaker timbre
                mfcc_std,               # 20 dims: timbre variability
                [pitch_mean],           # 1 dim: average pitch
                [pitch_std],            # 1 dim: pitch variability
                [voiced_ratio],         # 1 dim: how much is voiced
                [centroid_mean / 1000], # 1 dim: spectral brightness (normalized)
                [zcr_mean * 10],        # 1 dim: noisiness (scaled)
                [rms_mean * 100],       # 1 dim: energy (scaled)
            ])

            return feature_vector, median_pitch

        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            n_features = self.n_mfcc * 2 + 6
            return np.zeros(n_features), 0.0

    def diarize_segments(
        self,
        audio_segments: List[np.ndarray],
        sample_rate: int = 16000
    ) -> List[str]:
        """
        Perform speaker diarization on audio segments.

        Uses relative pitch comparison: the cluster with higher median pitch
        is labeled "child", and the lower one is labeled "adult".
        """
        if not audio_segments:
            return []

        logger.info(f"Diarizing {len(audio_segments)} segments...")

        # Extract features from all segments
        features_list = []
        pitch_list = []

        for i, audio in enumerate(audio_segments):
            if len(audio) < sample_rate * 0.1:  # Skip very short segments
                n_features = self.n_mfcc * 2 + 6
                features_list.append(np.zeros(n_features))
                pitch_list.append(0.0)
                continue

            feature_vector, median_pitch = self.extract_features(audio, sample_rate)
            features_list.append(feature_vector)
            pitch_list.append(median_pitch)

        features = np.array(features_list)
        pitches = np.array(pitch_list)

        # Normalize features
        features_normalized = self.scaler.fit_transform(features)

        # Cluster speakers using K-means
        self.kmeans = KMeans(
            n_clusters=self.n_speakers, random_state=42, n_init=10
        )
        cluster_labels = self.kmeans.fit_predict(features_normalized)

        # Calculate median pitch per cluster (more robust than mean)
        cluster_pitches = {}
        for cluster_id in range(self.n_speakers):
            mask = cluster_labels == cluster_id
            cluster_pitch_values = pitches[mask]
            valid_pitches = cluster_pitch_values[cluster_pitch_values > 0]
            if len(valid_pitches) > 0:
                cluster_pitches[cluster_id] = float(np.median(valid_pitches))
            else:
                cluster_pitches[cluster_id] = 0.0

        logger.info(f"Cluster median pitches: {cluster_pitches}")

        # Count segments per cluster
        for cid in range(self.n_speakers):
            count = int(np.sum(cluster_labels == cid))
            logger.info(f"  Cluster {cid}: {count} segments, "
                        f"median pitch = {cluster_pitches[cid]:.1f} Hz")

        # Relative comparison: higher pitch cluster = child, lower = adult
        valid_clusters = {k: v for k, v in cluster_pitches.items() if v > 0}

        if len(valid_clusters) >= 2:
            # Sort clusters by pitch: highest pitch = child
            sorted_clusters = sorted(
                valid_clusters.items(), key=lambda x: x[1], reverse=True
            )
            child_cluster = sorted_clusters[0][0]
            adult_cluster = sorted_clusters[1][0]

            pitch_diff = sorted_clusters[0][1] - sorted_clusters[1][1]
            logger.info(
                f"Speaker assignment (relative): "
                f"child=cluster {child_cluster} ({sorted_clusters[0][1]:.1f} Hz), "
                f"adult=cluster {adult_cluster} ({sorted_clusters[1][1]:.1f} Hz), "
                f"pitch diff={pitch_diff:.1f} Hz"
            )

            # If pitch difference is very small (<15 Hz), warn that
            # classification may be unreliable
            if pitch_diff < 15:
                logger.warning(
                    f"Pitch difference between clusters is small "
                    f"({pitch_diff:.1f} Hz). Speaker classification "
                    f"may be less reliable."
                )

            cluster_to_speaker = {
                child_cluster: "child",
                adult_cluster: "adult"
            }
            # Handle any additional clusters
            for cid in range(self.n_speakers):
                if cid not in cluster_to_speaker:
                    cluster_to_speaker[cid] = f"SPEAKER_{cid}"
        else:
            # Fallback: use fixed threshold if relative comparison not possible
            logger.warning("Cannot do relative comparison, using fixed threshold")
            cluster_to_speaker = {}
            for cid, pitch in cluster_pitches.items():
                if pitch > self.child_pitch_threshold:
                    cluster_to_speaker[cid] = "child"
                elif pitch > 0:
                    cluster_to_speaker[cid] = "adult"
                else:
                    cluster_to_speaker[cid] = f"SPEAKER_{cid}"

        # Assign speaker labels
        speaker_labels = [
            cluster_to_speaker[cid] for cid in cluster_labels
        ]

        # Log distribution
        child_count = sum(1 for s in speaker_labels if s == "child")
        adult_count = sum(1 for s in speaker_labels if s == "adult")
        logger.info(
            f"Diarization complete: {child_count} child segments, "
            f"{adult_count} adult segments"
        )

        return speaker_labels

    def diarize_with_timestamps(
        self,
        audio_segments: List[Tuple[np.ndarray, float, float]],
        sample_rate: int = 16000
    ) -> List[Dict]:
        """
        Diarize segments with timestamp information.
        """
        audio_only = [seg[0] for seg in audio_segments]
        speaker_labels = self.diarize_segments(audio_only, sample_rate)

        results = []
        for (audio, start, end), speaker in zip(audio_segments, speaker_labels):
            results.append({
                'start': start,
                'end': end,
                'speaker': speaker
            })

        return results


def create_simple_diarization(config: dict) -> SimpleDiarization:
    """Factory function to create simple diarization from config."""
    diar_config = config.get('simple_diarization', {})

    model = SimpleDiarization(
        n_speakers=diar_config.get('n_speakers', 2),
        n_mfcc=diar_config.get('n_mfcc', 20),
        child_pitch_threshold=diar_config.get('child_pitch_threshold', 250.0)
    )

    return model

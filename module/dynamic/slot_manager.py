from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import SlotManagerConfig

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


class SlotState(str, Enum):
    ACTIVE = "active"
    OCCLUDED = "occluded"
    INACTIVE = "inactive"


@dataclass
class DetectedSlot:
    position: np.ndarray
    visual_code: int
    confidence: float = 1.0

    @classmethod
    def from_mapping(cls, payload: dict) -> "DetectedSlot":
        position = payload.get("S_p", payload.get("position"))
        visual_code = payload.get("S_v_idx", payload.get("visual_code"))
        confidence = payload.get("confidence", 1.0)
        if position is None or visual_code is None:
            raise ValueError("Detected slot payload must include position/S_p and visual_code/S_v_idx.")
        return cls(
            position=np.asarray(position, dtype=np.float32),
            visual_code=int(visual_code),
            confidence=float(confidence),
        )


@dataclass
class TrackedSlot:
    slot_id: int
    position: np.ndarray
    visual_code: int
    state: SlotState = SlotState.ACTIVE
    age: int = 0
    occlusion_count: int = 0
    confidence: float = 1.0
    history: List[np.ndarray] = field(default_factory=list)


class SlotManager:
    def __init__(self, config: Optional[SlotManagerConfig] = None) -> None:
        self.config = config or SlotManagerConfig()
        self.next_id = 0
        self.active_slots: List[TrackedSlot] = []

    def predict_dynamics(self, slot: TrackedSlot, dt: float = 1.0) -> np.ndarray:
        cx, cy, vx, vy = slot.position
        return np.asarray(
            [
                cx + vx * dt,
                cy + vy * dt,
                vx * self.config.velocity_decay,
                vy * self.config.velocity_decay,
            ],
            dtype=np.float32,
        )

    def update_dynamics(self, slot: TrackedSlot, measurement: np.ndarray) -> None:
        predicted = self.predict_dynamics(slot)
        alpha = self.config.measurement_alpha
        cx_new = alpha * measurement[0] + (1.0 - alpha) * predicted[0]
        cy_new = alpha * measurement[1] + (1.0 - alpha) * predicted[1]
        vx_measured = measurement[0] - slot.position[0]
        vy_measured = measurement[1] - slot.position[1]
        slot.position = np.asarray(
            [
                cx_new,
                cy_new,
                0.8 * slot.position[2] + 0.2 * vx_measured,
                0.8 * slot.position[3] + 0.2 * vy_measured,
            ],
            dtype=np.float32,
        )

    def data_association(
        self,
        current_slots: Sequence[DetectedSlot],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not self.active_slots:
            return [], [], list(range(len(current_slots)))

        if not current_slots:
            return [], list(range(len(self.active_slots))), []

        cost_matrix = np.zeros((len(self.active_slots), len(current_slots)), dtype=np.float32)

        for track_idx, track in enumerate(self.active_slots):
            predicted_pos = self.predict_dynamics(track)[:2]
            for det_idx, detection in enumerate(current_slots):
                pos_dist = np.linalg.norm(predicted_pos - detection.position[:2])
                pos_cost = pos_dist / max(self.config.position_threshold, 1e-6)
                visual_cost = 0.0 if track.visual_code == detection.visual_code else 1.0
                cost_matrix[track_idx, det_idx] = pos_cost + 0.5 * visual_cost

        row_indices, col_indices = self._assign(cost_matrix)

        matches: List[Tuple[int, int]] = []
        unmatched_tracks = list(range(len(self.active_slots)))
        unmatched_dets = list(range(len(current_slots)))

        for row_idx, col_idx in zip(row_indices, col_indices):
            if cost_matrix[row_idx, col_idx] < self.config.cost_threshold:
                matches.append((row_idx, col_idx))
                unmatched_tracks.remove(row_idx)
                unmatched_dets.remove(col_idx)

        return matches, unmatched_tracks, unmatched_dets

    def step(self, detected_slots: Iterable[DetectedSlot | dict], global_context: Optional[dict] = None) -> List[dict]:
        parsed_slots = [
            detection if isinstance(detection, DetectedSlot) else DetectedSlot.from_mapping(detection)
            for detection in detected_slots
        ]
        matches, unmatched_tracks, unmatched_dets = self.data_association(parsed_slots)

        for track_idx, det_idx in matches:
            track = self.active_slots[track_idx]
            detection = parsed_slots[det_idx]
            self.update_dynamics(track, detection.position[:2])
            track.visual_code = detection.visual_code
            track.state = SlotState.ACTIVE
            track.occlusion_count = 0
            track.confidence = min(1.0, max(track.confidence, detection.confidence) + 0.1)
            track.age += 1
            track.history.append(track.position.copy())

        for track_idx in unmatched_tracks:
            track = self.active_slots[track_idx]
            track.state = SlotState.OCCLUDED
            track.occlusion_count += 1
            track.position = self.predict_dynamics(track)
            track.confidence *= 0.9
            track.history.append(track.position.copy())
            if not self.should_maintain(track, global_context):
                track.state = SlotState.INACTIVE

        for det_idx in unmatched_dets:
            detection = parsed_slots[det_idx]
            new_track = TrackedSlot(
                slot_id=self.next_id,
                position=np.asarray(detection.position, dtype=np.float32),
                visual_code=detection.visual_code,
                confidence=detection.confidence,
            )
            new_track.history.append(new_track.position.copy())
            self.active_slots.append(new_track)
            self.next_id += 1

        self.active_slots = [slot for slot in self.active_slots if slot.state != SlotState.INACTIVE]
        return self.get_output()

    def should_maintain(self, slot: TrackedSlot, global_context: Optional[dict]) -> bool:
        if slot.occlusion_count > self.config.max_occlusion:
            return False

        if slot.age < 5 and slot.occlusion_count > 3:
            return False

        if slot.confidence < self.config.min_confidence:
            return False

        if global_context and global_context.get("crowdedness", 0.0) > 0.9:
            return False

        cx, cy = slot.position[:2]
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            return False

        return True

    def get_output(self) -> List[dict]:
        outputs = []
        for slot in self.active_slots:
            outputs.append(
                {
                    "id": slot.slot_id,
                    "state": slot.state.value,
                    "position": slot.position.copy(),
                    "visual_code": slot.visual_code,
                    "confidence": slot.confidence,
                    "age": slot.age,
                }
            )
        return outputs

    def _assign(self, cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if linear_sum_assignment is not None:
            return linear_sum_assignment(cost_matrix)

        pairs = []
        used_rows = set()
        used_cols = set()
        flat_indices = np.argsort(cost_matrix, axis=None)
        rows, cols = np.unravel_index(flat_indices, cost_matrix.shape)
        for row_idx, col_idx in zip(rows.tolist(), cols.tolist()):
            if row_idx in used_rows or col_idx in used_cols:
                continue
            used_rows.add(row_idx)
            used_cols.add(col_idx)
            pairs.append((row_idx, col_idx))
        if not pairs:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
        pair_array = np.asarray(pairs, dtype=np.int64)
        return pair_array[:, 0], pair_array[:, 1]

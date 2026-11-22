"""Central deployment and recognition settings."""
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class Settings:
    dpi: int = 250
    orientation_side: int = 1700
    confidence: float = .94
    orientation_margin: float = 2.0
    engine_version: str = 'local-multievidence-1.2.0'
    card_width: int = 1500

    def signature(self) -> dict:
        return asdict(self)

# Relative to a detected, rectified table, never the PDF page.
MEMBER_ROIS = {
    'name': (.205, .005, .555, .112),
    'relation': (.72, .005, .995, .12),
    'id_number': (.20, .48, .575, .64),
    'registration_date': (.59, 1.005, 1.01, 1.15),
}

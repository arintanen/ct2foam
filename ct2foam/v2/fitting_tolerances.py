"""Tolerance configuration for NASA7 coefficient validation."""

from dataclasses import dataclass


@dataclass
class FittingTolerances:
    """Tolerance settings for NASA7 coefficient validation checks."""

    # Tmid matching tolerance (relative)
    tmid_rel_tol: float = 1e-12

    # Continuity check tolerances
    continuity_cp_tol: float = 1e-6
    continuity_cpdT_tol: float = 0.01  # Loose tolerance per old implementation
    continuity_h_tol: float = 1e-6
    continuity_s_tol: float = 1e-6

    # Consistency check tolerance (L2 norm, absolute)
    consistency_abs_tol: float = 1e-6  # Match old default

    @classmethod
    def default(cls):
        """Default tolerances matching old implementation."""
        return cls()

    @classmethod
    def strict(cls):
        """Stricter tolerances for high-precision mechanisms."""
        return cls(
            tmid_rel_tol=1e-14,
            continuity_cp_tol=1e-8,
            continuity_cpdT_tol=1e-6,
            continuity_h_tol=1e-8,
            continuity_s_tol=1e-8,
            consistency_abs_tol=1e-8,
        )

    @classmethod
    def relaxed(cls):
        """Relaxed tolerances for problematic mechanisms."""
        return cls(
            tmid_rel_tol=1e-10,
            continuity_cp_tol=1e-4,
            continuity_cpdT_tol=0.1,
            continuity_h_tol=1e-4,
            continuity_s_tol=1e-4,
            consistency_abs_tol=0.1,
        )

"""Pydantic models for mechanism genome (LLM output validation and JSON schema)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class MechanismModel(BaseModel):
    """
    Mechanism genome schema. Use model_validate(json) to parse LLM output;
    use model_dump() to get a dict for the simulator/evolution.
    """

    # Info policy
    info_mode: Literal["none", "coarse_bins", "exact"] = "none"
    info_bins: dict[str, float] | None = Field(default=None, description="Required when info_mode is coarse_bins")
    risk_labeling: Literal["hidden", "shown"] | None = None

    # Gating
    gating_mode: Literal["always_admit", "threshold", "probabilistic", "capacity_based"] = "always_admit"
    gating_threshold: float | None = Field(default=None, ge=0.0, le=3.0)
    capacity_load: float | None = Field(default=None, ge=0.0, le=1.0)

    # Priority
    service_rule: Literal["fifo", "severity_priority", "hybrid"] = "fifo"
    hybrid_a: float | None = Field(default=None, ge=0.0, le=5.0)
    hybrid_b: float | None = Field(default=None, ge=0.0, le=1.0)

    # Redirect/exit
    redirect_low_risk: bool = False
    redirect_threshold: float | None = Field(default=None, ge=0.0, le=3.0)
    reneging_enabled: bool = True
    reneging_model: dict[str, Any] | None = None

    @model_validator(mode="after")
    def require_conditional_fields(self) -> "MechanismModel":
        if self.info_mode == "coarse_bins" and not self.info_bins:
            self.info_bins = {"lt_30": 15, "30_90": 60, "gt_90": 120}
        if self.service_rule == "hybrid":
            if self.hybrid_a is None:
                self.hybrid_a = 1.0
            if self.hybrid_b is None:
                self.hybrid_b = 0.1
        if self.gating_mode == "threshold" and self.gating_threshold is None:
            self.gating_threshold = 0.5
        if self.gating_mode == "capacity_based" and self.capacity_load is None:
            self.capacity_load = 0.8
        return self

    def to_mechanism_dict(self) -> dict[str, Any]:
        """Export to dict with only non-None fields and correct keys for simulator."""
        d: dict[str, Any] = {
            "info_mode": self.info_mode,
            "gating_mode": self.gating_mode,
            "service_rule": self.service_rule,
            "redirect_low_risk": self.redirect_low_risk,
            "reneging_enabled": self.reneging_enabled,
        }
        if self.risk_labeling is not None:
            d["risk_labeling"] = self.risk_labeling
        if self.info_bins is not None:
            d["info_bins"] = self.info_bins
        if self.gating_threshold is not None:
            d["gating_threshold"] = self.gating_threshold
        if self.capacity_load is not None:
            d["capacity_load"] = self.capacity_load
        if self.hybrid_a is not None:
            d["hybrid_a"] = self.hybrid_a
        if self.hybrid_b is not None:
            d["hybrid_b"] = self.hybrid_b
        if self.redirect_threshold is not None:
            d["redirect_threshold"] = self.redirect_threshold
        if self.reneging_model is not None:
            d["reneging_model"] = self.reneging_model
        return d


def mechanism_json_schema_for_llm() -> dict[str, Any]:
    """Return JSON schema for the LLM prompt so it knows exact expected shape."""
    return MechanismModel.model_json_schema()

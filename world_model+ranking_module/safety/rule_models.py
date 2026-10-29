"""
Safeguard Agent Models
Pydantic models for safety assessment input/output.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class RiskLevel(str, Enum):
    """Risk assessment levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class AssessmentStatus(str, Enum):
    """Assessment result status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    REVIEW = "review"


class RiskFactor(BaseModel):
    """Individual risk factor identified"""
    factor: str = Field(..., description="Name of the risk factor")
    category: str = Field(
        ...,
        description="Category: medical, environmental, nutritional, exercise"
    )
    severity: RiskLevel = Field(..., description="Risk severity")
    description: str = Field(..., description="Detailed description")
    recommendation: str = Field(..., description="How to mitigate")


class SafetyCheck(BaseModel):
    """Individual safety check result"""
    check_name: str = Field(..., description="Name of the check")
    passed: bool = Field(..., description="Whether the check passed")
    message: str = Field(..., description="Result message")
    severity: Optional[RiskLevel] = Field(
        None,
        description="Severity if failed"
    )



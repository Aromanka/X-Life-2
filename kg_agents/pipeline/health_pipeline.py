"""
Health Pipeline
Orchestrates diet and exercise generation with safety assessment.
"""
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from agents.diet import generate_diet_candidates as diet_gen
from agents.diet.models import DietRecommendation
# from agents.exercise.generator import generate_exercise_candidates as exercise_gen
from agents.exercise.models import ExercisePlan
from agents.safeguard.assessor import assess_plan_safety, SafeguardAgent
from agents.safeguard.models import (
    SafetyAssessment, CombinedAssessment, SafeguardInput
)


# ================= Pipeline Input/Output =================

@dataclass
class PipelineInput:
    """Input for the health pipeline"""
    user_metadata: Dict[str, Any]
    environment: Dict[str, Any] = None
    user_requirement: Dict[str, Any] = None
    num_candidates: int = 3
    diet_only: bool = False
    exercise_only: bool = False


@dataclass
class PipelineOutput:
    """Output from the health pipeline"""
    diet_candidates: List[Dict[str, Any]] = None
    exercise_candidates: List[Dict[str, Any]] = None
    diet_assessments: Dict[int, SafetyAssessment] = None
    exercise_assessments: Dict[int, SafetyAssessment] = None
    combined_assessment: Dict[str, Any] = None
    generated_at: datetime = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "diet_candidates": self.diet_candidates,
            "exercise_candidates": self.exercise_candidates,
            "diet_assessments": {
                k: v.model_dump() for k, v in (self.diet_assessments or {}).items()
            },
            "exercise_assessments": {
                k: v.model_dump() for k, v in (self.exercise_assessments or {}).items()
            },
            "combined_assessment": self.combined_assessment,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None
        }


# ================= Health Pipeline =================

class HealthPlanPipeline:
    """
    Main pipeline for health plan generation.

    Flow:
    1. Generate diet candidates (optional)
    2. Generate exercise candidates (optional)
    3. Assess all candidates for safety
    4. Filter and sort by safety score
    5. Return combined results
    """

    def __init__(self):
        self.safeguard = SafeguardAgent()

    def generate(
        self,
        user_metadata: Dict[str, Any],
        environment: Dict[str, Any] = None,
        user_requirement: Dict[str, Any] = None,
        num_candidates: int = 3,
        diet_only: bool = False,
        exercise_only: bool = False
    ) -> PipelineOutput:
        """
        Generate health plans with safety assessment.

        Args:
            user_metadata: User physiological data
            environment: Environmental context
            user_requirement: User goals
            num_candidates: Number of candidates per type
            diet_only: Generate only diet plans
            exercise_only: Generate only exercise plans

        Returns:
            PipelineOutput with candidates and assessments
        """
        # Initialize output
        output = PipelineOutput(
            diet_candidates=[],
            exercise_candidates=[],
            diet_assessments={},
            exercise_assessments={},
            generated_at=datetime.now()
        )

        # Set defaults
        env = environment or {}
        req = user_requirement or {}

        # Generate diet candidates
        if not exercise_only:
            output.diet_candidates = self._generate_diet_candidates(
                user_metadata, env, req, num_candidates
            )

        # Generate exercise candidates
        if not diet_only:
            output.exercise_candidates = self._generate_exercise_candidates(
                user_metadata, env, req, num_candidates
            )

        # Assess safety for diet plans
        if output.diet_candidates:
            output.diet_assessments = self._assess_diet_candidates(
                output.diet_candidates, user_metadata, env
            )

        # Assess safety for exercise plans
        if output.exercise_candidates:
            output.exercise_assessments = self._assess_exercise_candidates(
                output.exercise_candidates, user_metadata, env
            )

        # Combined assessment
        output.combined_assessment = self._combined_assessment(
            output.diet_assessments or {},
            output.exercise_assessments or {}
        )

        return output

    def _generate_diet_candidates(
        self,
        user_metadata: Dict[str, Any],
        environment: Dict[str, Any],
        requirement: Dict[str, Any],
        num_candidates: int
    ) -> List[Dict[str, Any]]:
        """Generate diet candidates"""
        try:
            candidates = diet_gen(
                user_metadata=user_metadata,
                environment=environment,
                user_requirement=requirement,
                num_candidates=num_candidates
            )
            return [c.model_dump() for c in candidates]
        except Exception as e:
            print(f"[Pipeline] Diet generation failed: {e}")
            return []

    def _generate_exercise_candidates(
        self,
        user_metadata: Dict[str, Any],
        environment: Dict[str, Any],
        requirement: Dict[str, Any],
        num_candidates: int
    ) -> List[Dict[str, Any]]:
        """Generate exercise candidates"""
        try:
            # candidates = exercise_gen(
            #     user_metadata=user_metadata,
            #     environment=environment,
            #     user_requirement=requirement,
            #     num_candidates=num_candidates
            # )
            # return [c.model_dump() for c in candidates]
            pass
        except Exception as e:
            print(f"[Pipeline] Exercise generation failed: {e}")
            return []

    def _assess_diet_candidates(
        self,
        candidates: List[Dict[str, Any]],
        user_metadata: Dict[str, Any],
        environment: Dict[str, Any]
    ) -> Dict[int, SafetyAssessment]:
        """Assess diet candidates"""
        assessments = {}
        for plan in candidates:
            assessment = self.safeguard.assess(
                plan=plan,
                plan_type="diet",
                user_metadata=user_metadata,
                environment=environment
            )
            assessments[plan.get("id", 0)] = assessment
        return assessments

    def _assess_exercise_candidates(
        self,
        candidates: List[Dict[str, Any]],
        user_metadata: Dict[str, Any],
        environment: Dict[str, Any]
    ) -> Dict[int, SafetyAssessment]:
        """Assess exercise candidates"""
        assessments = {}
        for plan in candidates:
            assessment = self.safeguard.assess(
                plan=plan,
                plan_type="exercise",
                user_metadata=user_metadata,
                environment=environment
            )
            assessments[plan.get("id", 0)] = assessment
        return assessments

    def _combined_assessment(
        self,
        diet_assessments: Dict[int, SafetyAssessment],
        exercise_assessments: Dict[int, SafetyAssessment]
    ) -> Dict[str, Any]:
        """Generate combined assessment"""
        all_assessments = list(diet_assessments.values()) + list(exercise_assessments.values())

        if not all_assessments:
            return {
                "overall_score": 100,
                "is_safe": True,
                "recommendations": ["No candidates generated"]
            }

        # Calculate overall score
        overall_score = sum(a.score for a in all_assessments) // len(all_assessments)
        is_safe = all(a.is_safe for a in all_assessments)

        # Generate recommendations
        recommendations = []
        high_risk = [a for a in all_assessments if a.risk_level.value in ["high", "very_high"]]

        if high_risk:
            recommendations.append("Some plans have high risk. Review safety notes carefully.")

        for a in all_assessments[:3]:  # Top 3 recommendations
            recommendations.extend(a.recommendations[:2])

        # Remove duplicates while preserving order
        seen = set()
        unique_recs = []
        for r in recommendations:
            if r not in seen:
                seen.add(r)
                unique_recs.append(r)

        return {
            "overall_score": overall_score,
            "is_safe": is_safe,
            "risk_level": "low" if overall_score >= 80 else "moderate" if overall_score >= 60 else "high",
            "recommendations": unique_recs[:5],
            "total_assessed": len(all_assessments)
        }

    def filter_safe_candidates(
        self,
        output: PipelineOutput,
        min_score: int = 60
    ) -> PipelineOutput:
        """
        Filter candidates by minimum safety score.

        Args:
            output: PipelineOutput
            min_score: Minimum safety score (0-100)

        Returns:
            Filtered PipelineOutput
        """
        # Filter diet
        safe_diet_ids = [
            plan_id for plan_id, assessment in (output.diet_assessments or {}).items()
            if assessment.score >= min_score
        ]
        output.diet_candidates = [
            c for c in output.diet_candidates
            if c.get("id") in safe_diet_ids
        ]

        # Filter exercise
        safe_exercise_ids = [
            plan_id for plan_id, assessment in (output.exercise_assessments or {}).items()
            if assessment.score >= min_score
        ]
        output.exercise_candidates = [
            c for c in output.exercise_candidates
            if c.get("id") in safe_exercise_ids
        ]

        return output

    def sort_by_safety(
        self,
        output: PipelineOutput,
        ascending: bool = False
    ) -> PipelineOutput:
        """
        Sort candidates by safety score.

        Args:
            output: PipelineOutput
            ascending: Sort ascending (False = highest first)

        Returns:
            Sorted PipelineOutput
        """
        # Sort diet
        if output.diet_candidates and output.diet_assessments:
            output.diet_candidates.sort(
                key=lambda c: output.diet_assessments.get(c.get("id"), SafetyAssessment(score=0)).score,
                reverse=not ascending
            )

        # Sort exercise
        if output.exercise_candidates and output.exercise_assessments:
            output.exercise_candidates.sort(
                key=lambda c: output.exercise_assessments.get(c.get("id"), SafetyAssessment(score=0)).score,
                reverse=not ascending
            )

        return output


# ================= Convenience Function =================

def generate_health_plans(
    user_metadata: Dict[str, Any],
    environment: Dict[str, Any] = None,
    user_requirement: Dict[str, Any] = None,
    num_candidates: int = 3,
    filter_safe: bool = True,
    min_score: int = 60
) -> Dict[str, Any]:
    """
    Convenience function to generate complete health plans.

    Args:
        user_metadata: User physiological data
        environment: Environmental context
        user_requirement: User goals
        num_candidates: Number of candidates per type
        filter_safe: Whether to filter to safe candidates only
        min_score: Minimum safety score for filtering

    Returns:
        Dictionary with diet_candidates, exercise_candidates, and assessments
    """
    pipeline = HealthPlanPipeline()

    output = pipeline.generate(
        user_metadata=user_metadata,
        environment=environment,
        user_requirement=user_requirement,
        num_candidates=num_candidates
    )

    if filter_safe:
        output = pipeline.filter_safe_candidates(output, min_score)

    return output.to_dict()

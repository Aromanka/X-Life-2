"""
Exercise Plan Parser
Parses and expands LLM-generated base exercise plans into multiple intensity variants.

Strategy: Generate variants with configurable scale factors.
"""
from typing import List, Dict, Any
from .models import ExerciseItem, ExerciseSession, ExercisePlan


class ExercisePlanParser:
    """
    Exercise plan parser and expander.

    Takes a base plan from LLM and expands it into multiple variants
    with configurable scale factors.
    """

    def __init__(self, num_variants: int = 3, min_scale: float = 0.7, max_scale: float = 1.3):
        # Store configuration
        self.num_variants = num_variants
        self.min_scale = min_scale
        self.max_scale = max_scale

        # Generate scale factors uniformly distributed between min_scale and max_scale
        if num_variants == 1:
            scale_factors = [(min_scale + max_scale) / 2]
        elif num_variants == 2:
            scale_factors = [min_scale, max_scale]
        else:
            step = (max_scale - min_scale) / (num_variants - 1)
            scale_factors = [min_scale + i * step for i in range(num_variants)]

        # Generate variant names: Variant_1, Variant_2, etc.
        self.variant_configs = [
            (f"Variant_{i+1}", round(factor, 3))
            for i, factor in enumerate(scale_factors)
        ]
        self.variants = {name: factor for name, factor in self.variant_configs}

        # Generate intensity mappings dynamically based on scale factors
        self.intensity_map = {}
        for variant_name, scale_factor in self.variants.items():
            self.intensity_map[variant_name] = self._build_intensity_map(scale_factor)

    def _build_intensity_map(self, scale_factor: float) -> Dict[str, str]:
        """
        Build intensity mapping based on scale factor.
        Lower scale -> lower intensity, Higher scale -> higher intensity.
        """
        # Base intensity map for scale_factor = 1.0 (Standard)
        base_map = {
            "very_high": "very_high",
            "high": "high",
            "moderate": "moderate",
            "low": "low"
        }

        if scale_factor < 1.0:
            # Lower scale: reduce intensity
            return {
                "very_high": "high",
                "high": "moderate",
                "moderate": "low",
                "low": "low"
            }
        elif scale_factor > 1.0:
            # Higher scale: increase intensity
            return {
                "very_high": "very_high",
                "high": "very_high",
                "moderate": "high",
                "low": "moderate"
            }
        else:
            # Scale factor = 1.0: keep base intensity
            return base_map

    def expand_plan(
        self,
        base_plan: ExercisePlan,
        variants: List[str] = None
    ) -> Dict[str, ExercisePlan]:
        """
        Expand a base plan into multiple intensity variants.

        Args:
            base_plan: ExercisePlan from LLM
            variants: Which variants to generate (default: all configured variants)

        Returns:
            Dict mapping variant name to expanded ExercisePlan
        """
        if variants is None:
            variants = [name for name, _ in self.variant_configs]

        result = {}

        for variant_name in variants:
            scale_factor = self.variants.get(variant_name, 1.0)
            expanded_plan = self._scale_plan(base_plan, scale_factor, variant_name)
            result[variant_name] = expanded_plan

        return result

    def _scale_plan(
        self,
        base_plan: ExercisePlan,
        scale_factor: float,
        variant_name: str
    ) -> ExercisePlan:
        """
        Scale a complete exercise plan according to rules.

        Args:
            base_plan: ExercisePlan to scale
            scale_factor: Multiplier (e.g., 0.7, 1.0, 1.3)
            variant_name: Which variant (for intensity mapping)

        Returns:
            ExercisePlan with scaled durations and adjusted intensities
        """
        scaled_sessions = {}

        for time_key, session in base_plan.sessions.items():
            scaled_session = self._scale_session(session, scale_factor, variant_name)
            scaled_sessions[time_key] = scaled_session

        # Calculate totals
        total_duration = sum(
            s.total_duration_minutes for s in scaled_sessions.values()
        )
        total_calories = sum(
            s.total_calories_burned for s in scaled_sessions.values()
        )

        return ExercisePlan(
            id=base_plan.id,
            title=f"{base_plan.title} ({variant_name})",
            meal_timing=base_plan.meal_timing,
            start_minutes_after_meal=base_plan.start_minutes_after_meal,
            sessions=scaled_sessions,
            total_duration_minutes=total_duration,
            total_calories_burned=total_calories,
            progression=base_plan.progression,
            reasoning=base_plan.reasoning,
            safety_notes=base_plan.safety_notes
        )

    def _scale_session(
        self,
        session: ExerciseSession,
        scale_factor: float,
        variant_name: str
    ) -> ExerciseSession:
        """
        Scale a single exercise session.

        Args:
            session: ExerciseSession to scale
            scale_factor: Duration multiplier
            variant_name: For intensity mapping

        Returns:
            ExerciseSession with scaled exercises
        """
        scaled_exercises = []
        intensity_map = self.intensity_map.get(variant_name, {})

        for exercise in session.exercises:
            # Scale duration
            new_duration = round(exercise.duration_minutes * scale_factor)
            new_duration = max(5, new_duration)  # Minimum 5 minutes per exercise

            # Scale calories proportionally
            if exercise.duration_minutes > 0:
                new_calories = round(
                    exercise.calories_burned * (new_duration / exercise.duration_minutes)
                )
            else:
                new_calories = exercise.calories_burned

            # Adjust intensity
            new_intensity = exercise.intensity
            if exercise.intensity.value in intensity_map:
                new_intensity = intensity_map[exercise.intensity.value]

            # Create scaled exercise
            scaled_exercise = ExerciseItem(
                name=exercise.name,
                exercise_type=exercise.exercise_type,
                duration_minutes=new_duration,
                intensity=new_intensity,
                MET=exercise.MET,
                calories_burned=new_calories,
                equipment=exercise.equipment,
                target_muscles=exercise.target_muscles,
                instructions=exercise.instructions,
                reason=exercise.reason,
                safety_notes=exercise.safety_notes
            )
            scaled_exercises.append(scaled_exercise)

        # Calculate session totals
        total_duration = sum(ex.duration_minutes for ex in scaled_exercises)
        total_calories = sum(ex.calories_burned for ex in scaled_exercises)

        # Determine overall intensity
        if scaled_exercises:
            intensities = [ex.intensity.value for ex in scaled_exercises]
            if "very_high" in intensities:
                overall = "very_high"
            elif "high" in intensities:
                overall = "high"
            elif "moderate" in intensities:
                overall = "moderate"
            else:
                overall = "low"
        else:
            overall = "low"

        return ExerciseSession(
            time_of_day=session.time_of_day,
            exercises=scaled_exercises,
            total_duration_minutes=total_duration,
            total_calories_burned=total_calories,
            overall_intensity=overall
        )

    def expand_single_plan(
        self,
        base_plan: ExercisePlan
    ) -> Dict[str, ExercisePlan]:
        """
        Convenience: Expand a single plan into all variants.

        Returns:
            Dict mapping variant name to scaled ExercisePlan
        """
        return self.expand_plan(base_plan)


# Convenience function
def expand_exercise_plan(
    base_plan: ExercisePlan
) -> Dict[str, ExercisePlan]:
    """
    Quick helper to expand an exercise plan.

    Args:
        base_plan: ExercisePlan from LLM

    Returns:
        Dict with Variant_1, Variant_2, ... variants
    """
    parser = ExercisePlanParser()
    return parser.expand_plan(base_plan)

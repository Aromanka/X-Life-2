from typing import List, Dict, Any, Optional
from pydantic import AliasChoices, BaseModel, Field
from enum import Enum


class ExerciseType(str, Enum):
    """Types of exercise"""
    CARDIO = "cardio"
    STRENGTH = "strength"
    FLEXIBILITY = "flexibility"
    BALANCE = "balance"
    HIIT = "hiit"


class IntensityLevel(str, Enum):
    """Exercise intensity levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TimeOfDay(str, Enum):
    """Time of day for exercise"""
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    ANY = "any"


class ExerciseItem(BaseModel):
    """A single exercise in a plan"""
    name: str = Field(..., description="Name of the exercise")
    exercise_type: ExerciseType = Field(..., description="Type of exercise")
    duration_minutes: int = Field(
        ...,
        ge=1,
        le=180,
        description="Duration in minutes (1-180)"
    )
    intensity: IntensityLevel = Field(..., description="Exercise intensity")
    MET: Optional[float] = Field(
        None,
        validation_alias=AliasChoices("MET", "met"),
        description="Metabolic equivalent of task"
    )
    calories_burned: int = Field(
        ...,
        ge=0,
        le=1000,
        description="Estimated calories burned (0-1000)"
    )
    equipment: List[str] = Field(
        default_factory=list,
        description="Required equipment"
    )
    target_muscles: List[str] = Field(
        default_factory=list,
        description="Target muscle groups"
    )
    instructions: List[str] = Field(
        default_factory=list,
        description="Exercise instructions"
    )
    reason: Optional[str] = Field(
        None,
        description="Why this exercise is suitable for the user"
    )
    safety_notes: List[str] = Field(
        default_factory=list,
        description="Safety considerations"
    )


class ExerciseSession(BaseModel):
    """A session of exercises (e.g., morning, afternoon)"""
    time_of_day: TimeOfDay = Field(..., description="When to perform")
    exercises: List[ExerciseItem] = Field(..., description="Exercises in this session")
    total_duration_minutes: int = Field(
        ...,
        ge=0,
        le=300,
        description="Total duration in minutes (0-300)"
    )
    total_calories_burned: int = Field(
        ...,
        ge=0,
        le=2000,
        description="Total calories burned (0-2000)"
    )
    overall_intensity: IntensityLevel = Field(..., description="Session intensity")


class MealTiming(str, Enum):
    """When to exercise relative to meals"""
    BEFORE_BREAKFAST = "before_breakfast"
    AFTER_BREAKFAST = "after_breakfast"
    BEFORE_LUNCH = "before_lunch"
    AFTER_LUNCH = "after_lunch"
    BEFORE_DINNER = "before_dinner"
    AFTER_DINNER = "after_dinner"
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"


class ExercisePlan(BaseModel):
    """Complete exercise plan for one candidate"""
    id: int = Field(..., description="Candidate ID", ge=1)
    title: str = Field(..., description="Plan title")
    meal_timing: MealTiming = Field(
        ...,
        description="When to exercise relative to meals"
    )
    start_minutes_after_meal: Optional[float] = Field(
        None,
        description="Exercise start in minutes relative to meal start"
    )
    sessions: Dict[str, ExerciseSession] = Field(
        ...,
        description="Exercise sessions keyed by time of day"
    )
    total_duration_minutes: int = Field(
        ...,
        ge=0,
        le=480,
        description="Total exercise duration in minutes (0-480)"
    )
    total_calories_burned: int = Field(
        ...,
        ge=0,
        le=3000,
        description="Total calories burned (0-3000)"
    )
    progression: Optional[str] = Field(
        None,
        description="How to progress over time"
    )
    reasoning: Optional[str] = Field(
        None,
        description="Overall reasoning for this plan"
    )
    safety_notes: List[str] = Field(
        default_factory=list,
        description="Safety considerations"
    )


class ExerciseCandidatesResponse(BaseModel):
    """Response containing multiple exercise candidates"""
    candidates: List[ExercisePlan] = Field(
        ...,
        description="List of exercise candidates"
    )
    target_calories_burned: int = Field(
        ...,
        description="Target daily calories to burn"
    )
    user_conditions: List[str] = Field(
        ...,
        description="User's health conditions considered"
    )
    fitness_level: str = Field(..., description="User's fitness level")
    sampling_strategy: str = Field(
        ...,
        description="Strategy used for generation"
    )
    generation_notes: str = Field(
        ...,
        description="Additional notes about generation"
    )


class ExerciseAgentInput(BaseModel):
    """Input structure for exercise agent"""
    user_metadata: Dict[str, Any] = Field(
        ...,
        description="User physiological data"
    )
    environment: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environmental context (weather, time)"
    )
    user_requirement: Dict[str, Any] = Field(
        default_factory=dict,
        description="User goals and preferences"
    )
    num_candidates: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of candidates to generate"
    )

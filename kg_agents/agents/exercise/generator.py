import json
import os
from typing import List, Dict, Any, Optional
from agents.base import BaseAgent, DietAgentMixin, ExerciseAgentMixin
from agents.exercise.models import (
    ExerciseType, IntensityLevel, TimeOfDay, MealTiming,
    ExerciseItem, ExerciseSession, ExercisePlan,
    ExerciseCandidatesResponse, ExerciseAgentInput
)
from agents.exercise.parser_var import ExercisePlanParser
from core.llm import get_llm
from core.llm.utils import parse_json_response
from core.neo4j import get_kg_query
import random
from agents.exercise.config import *
from core.prompts import (
    GET_EXERCISE_GENERATION_SYSTEM_PROMPT,
    build_exercise_prompt
)


MEAL_TIMING_OPTIONS = [
    "before_breakfast", "after_breakfast", "before_lunch", "after_lunch", "before_dinner", "after_dinner"
]

def build_exercise_constraint_prompt(
    primary_cardio: str = None,
    primary_strength: str = None,
    flexibility: str = None,
    excluded: List[str] = None,
    equipment: str = None,
    outdoor: bool = False,
    meal_timing: str = None
) -> str:
    """
    Build constraint prompt for mandatory exercise selection.

    Args:
        primary_cardio: Main cardio activity to include
        primary_strength: Main strength movement to include
        flexibility: Flexibility pose/flow to include
        excluded: List of exercises to exclude
        equipment: Required equipment to use
        outdoor: Whether to prioritize outdoor activities
        meal_timing: When to exercise relative to meals

    Returns:
        Formatted constraint string for LLM prompt
    """
    prompt_parts = []

    if meal_timing:
        timing_display = meal_timing.replace("_", " ").title()
        prompt_parts.append(f"- MEAL TIMING: {timing_display}")

    if primary_cardio:
        prompt_parts.append(f"- PRIMARY CARDIO: {primary_cardio}")

    if primary_strength:
        prompt_parts.append(f"- PRIMARY STRENGTH: {primary_strength}")

    if flexibility:
        prompt_parts.append(f"- FLEXIBILITY ELEMENT: {flexibility}")

    if equipment:
        prompt_parts.append(f"- EQUIPMENT REQUIRED: {equipment}")

    if outdoor:
        prompt_parts.append("- PRIORITIZE: Outdoor activities where possible")

    if excluded:
        prompt_parts.append(f"\n## EXCLUDED EXERCISES (DO NOT USE):")
        prompt_parts.append(f"- {', '.join(excluded)}")

    if prompt_parts:
        return "\n## EXERCISE CONSTRAINTS\n" + "\n".join(prompt_parts)

    return ""


class ExerciseAgent(BaseAgent, ExerciseAgentMixin):
    """Agent for generating exercise plan candidates"""

    def calculate_target_calories(
        self,
        weight_kg: float,
        goal: str = "maintenance",
        duration_minutes: int = None
    ) -> int:
        if duration_minutes is not None:
            goal_targets = {
                "weight_loss": 400,
                "muscle_building": 200,
                "cardio_improvement": 450,
                "flexibility": 100,
                "endurance": 400,
                "general_fitness": 250,
                "maintenance": 150
            }

            # Apply condition adjustments
            conditions = []
            if hasattr(self, '_input_meta'):
                conditions = self._input_meta.get("medical_conditions", [])

            # Reduce targets for certain conditions
            reduction_factor = 1.0
            for condition in conditions:
                if condition.lower() in ["heart_disease", "obesity", "arthritis"]:
                    reduction_factor = 0.75

            return int(goal_targets.get(goal, 250) * reduction_factor)
        else:
            # Calculate based on duration: ~5-8 kcal/min average for moderate exercise
            # Using simple MET-based estimation (MET = 4-8 for moderate exercise)
            # Calories = MET * weight(kg) * duration(min) / 60
            met_avg = 6  # Average MET for moderate exercise
            default_duration = 30  # Default 30 minutes if not specified
            return int(met_avg * weight_kg * default_duration / 60)

    def calculate_target_duration(
        self,
        fitness_level: str = "beginner",
        goal: str = "maintenance",
        duration_minutes: int = None
    ) -> int:
        if duration_minutes is not None:
            return duration_minutes

        # Base duration by fitness level (minutes) - used only if duration not provided
        base_durations = {
            "beginner": 20,
            "intermediate": 40,
            "advanced": 60
        }

        # Goal multipliers
        goal_multipliers = {
            "weight_loss": 1.0,
            "muscle_building": 0.85,
            "cardio_improvement": 1.1,
            "flexibility": 0.7,
            "endurance": 1.0,
            "general_fitness": 1.0,
            "maintenance": 0.9
        }

        base = base_durations.get(fitness_level.lower(), 30)
        multiplier = goal_multipliers.get(goal.lower(), 1.0)

        return int(base * multiplier)

    def calculate_target_weekly_frequency(
        self,
        fitness_level: str = "beginner",
        conditions: List[str] = None
    ) -> int:
        base_freq = {
            "beginner": 3,
            "intermediate": 4,
            "advanced": 5
        }

        # Condition restrictions
        condition_restrictions = {
            "heart_disease": -2,
            "obesity": -1,
            "arthritis": -1,
            "back_pain": 0
        }

        freq = base_freq.get(fitness_level.lower(), 3)

        if conditions:
            for cond in conditions:
                cond_lower = cond.lower()
                for key, adj in condition_restrictions.items():
                    if key in cond_lower or cond_lower in key:
                        freq = max(1, freq + adj)
                        break

        return max(1, min(7, freq))

    def get_agent_name(self) -> str:
        return "exercise"

    def get_input_type(self):
        return ExerciseAgentInput

    def get_output_type(self):
        return ExercisePlan

    def generate(
        self,
        input_data: Dict[str, Any],
        num_base_plans: int = 3,
        meal_timing: str = "",
        user_preference: str = None,
        use_vector: bool = True,  # GraphRAG: use vector search instead of keyword matching
        rag_topk: int = 3,
        kg_context: str = None,
        temperature: float = 0.7
    ) -> List[ExercisePlan]:
        # KG Format Version
        KG_FORMAT_VER = 3

        # Parse input
        input_obj = ExerciseAgentInput(**input_data)
        self._input_meta = input_obj.user_metadata  # Store for condition access

        user_meta = input_obj.user_metadata
        env = input_obj.environment
        requirement = input_obj.user_requirement

        # Extract key parameters
        conditions = user_meta.get("medical_conditions", [])
        fitness_level = user_meta.get("fitness_level", "beginner")
        weight = user_meta.get("weight_kg", 70)
        duration = requirement.get("duration", None)
        preferred_intensity = requirement.get("intensity", "moderate")

        # Calculate target metrics using instance methods
        target_calories = self.calculate_target_calories(weight, goal="maintenance", duration_minutes=duration)
        target_duration = self.calculate_target_duration(fitness_level, goal="maintenance", duration_minutes=duration)
        target_frequency = self.calculate_target_weekly_frequency(fitness_level, conditions)

        if kg_context is None:
            # Get knowledge graph context using mixin
            kg_context = ""
            if conditions:
                exercise_knowledge = self.query_exercise_knowledge(conditions, fitness_level)
                kg_context = self._format_kg_context(exercise_knowledge)

            # Query entity-based KG context when user_preference is provided
            if user_preference:
                entity_knowledge = self.query_exercise_by_entity(
                    user_preference,
                    use_vector_search=use_vector,
                    rag_topk=rag_topk,
                    kg_format_ver=KG_FORMAT_VER
                )
                entity_context = self._format_exercise_entity_kg_context(entity_knowledge, kg_format_ver=KG_FORMAT_VER)
                kg_context += entity_context
        else:
            pass

        # Get environment context
        weather = env.get("weather", {})
        season = env.get("time_context", {}).get("season", "any")

        # Build base user prompt
        # base_prompt = self._build_exercise_prompt(
        base_prompt = build_exercise_prompt(
            user_meta=user_meta,
            environment=env,
            requirement=requirement,
            kg_context=kg_context,
            target_calories=target_calories,
            target_duration=duration,
            target_intensity=preferred_intensity,
            target_frequency=target_frequency,
            user_preference=user_preference
        )

        # Generate candidates with mandatory exercise injection
        candidates = []
        used_combinations = set()

        for i in range(num_base_plans):
            if user_preference:
                primary_cardio = None
                primary_strength = None
                flexibility = None
                excluded = []
                equipment = None
            else:
                primary_cardio = random.choice(CARDIO_ACTIVITIES)
                primary_strength = random.choice(STRENGTH_MOVEMENTS)
                flexibility = random.choice(FLEXIBILITY_POSES)

            excluded = []
            if not user_preference and random.random() > 0.5:
                excluded = random.sample(COMMON_BORING_EXERCISES, k=random.randint(1, 2))

            equipment = None
            if not user_preference and random.random() > 0.7:
                equipment = random.choice(EQUIPMENT_OPTIONS)

            outdoor = False
            if not user_preference and weather.get("condition") in ["clear", "sunny"] and random.random() > 0.5:
                outdoor = True

            if not user_preference:
                combo_key = f"{meal_timing}-{primary_cardio}-{primary_strength}"
                # if not user_preference and combo_key in used_combinations and num_candidates < len(CARDIO_ACTIVITIES):
                #     primary_cardio = random.choice(CARDIO_ACTIVITIES)
                #     combo_key = f"{meal_timing}-{primary_cardio}-{primary_strength}"
                used_combinations.add(combo_key)

            constraint_prompt = build_exercise_constraint_prompt(
                primary_cardio=primary_cardio,
                primary_strength=primary_strength,
                flexibility=flexibility,
                excluded=excluded,
                equipment=equipment,
                outdoor=outdoor,
                meal_timing=meal_timing
            )

            full_prompt = base_prompt + "\n" + constraint_prompt

            candidate = self._generate_single_candidate(
                user_prompt=full_prompt,
                candidate_id=i + 1,
                fitness_level=fitness_level,
                weight=weight,
                temperature=temperature
                # strategy=strategy
            )
            if candidate:
                candidates.append(candidate)

        return candidates, kg_context

    def _format_kg_context(self, knowledge: List) -> str:
        """Format KG knowledge for prompt inclusion (matching diet agent pattern)"""
        if not knowledge:
            return ""

        parts = []

        maximum_inputs = 20
        if len(knowledge) > maximum_inputs:
            random.shuffle(knowledge)
            knowledge = knowledge[:maximum_inputs]

        for item in knowledge:
            entity_name = item.get("entity", "name")
            rel = item.get("rel", "relation")
            tail = item.get("tail", "name")
            condition = item.get("condition", "condition")

            part = "<{}, {}, {}> regarding {}".format(entity_name, rel, tail, condition)
            parts.append(part)

        return "#### Request based KG Guidelines:\n" + "\n".join(parts) + "\n"

    def _format_exercise_entity_kg_context(
        self, entity_knowledge: Dict, kg_format_ver: int = 2
    ) -> str:
        """Format entity-based KG knowledge for exercise prompt (matching diet agent pattern)"""
        if not entity_knowledge:
            return ""

        parts = []

        if kg_format_ver >= 3:
            # Simplified: uniform pattern for all relations
            matched_entities = entity_knowledge.get("matched_entities", [])
            relations = entity_knowledge.get("relations", [])

            # Format matched entities
            parts.append(f"Matched Entities: {', '.join(matched_entities)}")
            parts.append("")  # Empty line for separation

            # Format all relations uniformly: "- {head} {relation} {tail}"
            parts.append("## Knowledge Graph Relations")
            for rel in relations:
                head = rel.get("head", "")
                relation = rel.get("relation", "").replace("_", " ")
                tail = rel.get("tail", "")
                parts.append(f"- {head} {relation} {tail}")
        elif kg_format_ver == 1:
            if entity_knowledge.get("matched_entities"):
                entities = entity_knowledge["matched_entities"]
                parts.append(f"- Matched Entities from KG: {', '.join(set(entities))}")

            if entity_knowledge.get("entity_benefits"):
                benefits = entity_knowledge["entity_benefits"][:5]  # Limit to top 5
                unique_benefits = {}
                for b in benefits:
                    key = f"{b.get('entity', '')}-{b.get('benefit', '')}"
                    if key not in unique_benefits:
                        unique_benefits[key] = b
                if unique_benefits:
                    benefit_list = [f"{b.get('entity', '')} (has benefit of)/(is good for) {b.get('benefit', '')}" for b in unique_benefits.values()]
                    parts.append(f"- Exercise Benefits: {', '.join(benefit_list)}")

            if entity_knowledge.get("target_muscles"):
                muscles = entity_knowledge["target_muscles"][:5]  # Limit to top 5
                unique_muscles = {}
                for m in muscles:
                    key = f"{m.get('entity', '')}-{m.get('target', '')}"
                    if key not in unique_muscles:
                        unique_muscles[key] = m
                if unique_muscles:
                    muscle_list = [f"{m.get('entity', '')} targets {m.get('target', '')}" for m in unique_muscles.values()]
                    parts.append(f"- Target Muscles: {', '.join(muscle_list)}")

            if entity_knowledge.get("duration_recommendations"):
                durations = entity_knowledge["duration_recommendations"][:5]  # Limit to top 5
                unique_durations = {}
                for d in durations:
                    key = f"{d.get('entity', '')}-{d.get('duration', '')}"
                    if key not in unique_durations:
                        unique_durations[key] = d
                if unique_durations:
                    duration_list = [f"{d.get('entity', '')}: {d.get('duration', '')}" for d in unique_durations.values()]
                    parts.append(f"- Duration Recommendations: {', '.join(duration_list)}")

            if entity_knowledge.get("frequency_recommendations"):
                frequencies = entity_knowledge["frequency_recommendations"][:5]  # Limit to top 5
                unique_frequencies = {}
                for f in frequencies:
                    key = f"{f.get('entity', '')}-{f.get('frequency', '')}"
                    if key not in unique_frequencies:
                        unique_frequencies[key] = f
                if unique_frequencies:
                    freq_list = [f"{f.get('entity', '')}: {f.get('frequency', '')}" for f in unique_frequencies.values()]
                    parts.append(f"- Frequency Recommendations: {', '.join(freq_list)}")
        elif kg_format_ver == 2:
            # Organize by entities instead of by categories
            matched_entities = entity_knowledge.get("matched_entities", [])
            entity_benefits = entity_knowledge.get("entity_benefits", [])
            target_muscles = entity_knowledge.get("target_muscles", [])
            duration_recommendations = entity_knowledge.get("duration_recommendations", [])
            frequency_recommendations = entity_knowledge.get("frequency_recommendations", [])

            # Group relations by entity
            entity_relations = {}
            for entity in matched_entities:
                entity_relations[entity] = {
                    "benefits": [],
                    "target_muscles": [],
                    "durations": [],
                    "frequencies": []
                }

            # Populate benefits
            for b in entity_benefits:
                entity = b.get("entity", "")
                benefit = b.get("benefit", "")
                if entity in entity_relations and benefit:
                    entity_relations[entity]["benefits"].append(benefit)

            # Populate target muscles
            for m in target_muscles:
                entity = m.get("entity", "")
                target = m.get("target", "")
                if entity in entity_relations and target:
                    entity_relations[entity]["target_muscles"].append(target)

            # Populate duration recommendations
            for d in duration_recommendations:
                entity = d.get("entity", "")
                duration = d.get("duration", "")
                if entity in entity_relations and duration:
                    entity_relations[entity]["durations"].append(duration)

            # Populate frequency recommendations
            for f in frequency_recommendations:
                entity = f.get("entity", "")
                frequency = f.get("frequency", "")
                if entity in entity_relations and frequency:
                    entity_relations[entity]["frequencies"].append(frequency)

            # Format by entity
            parts.append(f"Matched Entities: {', '.join(matched_entities)}")
            parts.append("")  # Empty line for separation

            for entity in matched_entities:
                parts.append(f"### Entity: {entity}")
                relations = entity_relations[entity]

                if relations["benefits"]:
                    for benefit in relations["benefits"]:
                        parts.append(f"- {entity} has benefit of {benefit}")

                if relations["target_muscles"]:
                    for muscle in relations["target_muscles"]:
                        parts.append(f"- {entity} targets {muscle}")

                if relations["durations"]:
                    for duration in relations["durations"]:
                        parts.append(f"- {entity}: recommended duration {duration}")

                if relations["frequencies"]:
                    for freq in relations["frequencies"]:
                        parts.append(f"- {entity}: recommended frequency {freq}")

                parts.append("")  # Empty line between entities


        if parts:
            return "## Entity-Based KG Context\n" + "\n".join(parts) + "\n"
        return ""


    def _build_exercise_prompt(
        self,
        user_meta: Dict[str, Any],
        environment: Dict[str, Any],
        requirement: Dict[str, Any],
        kg_context: str,
        target_calories: int,
        target_duration: int,
        target_intensity: str,
        target_frequency: int,
        user_preference: str = None
    ) -> str:
        """Build the user prompt for exercise generation"""
        conditions = user_meta.get("medical_conditions", [])
        fitness_level = user_meta.get("fitness_level", "beginner")
        weight = user_meta.get("weight_kg", 70)
        goal = requirement.get("goal", "maintenance")

        prompt = f"""## TARGET TASK
Generate an exercise plan for the following user.
"""

        # User Preference at the TOP with HIGHEST PRIORITY
        if user_preference:
            prompt += f"""
### USER REQUEST (HIGHEST PRIORITY):
The user strictly explicitly wants: "{user_preference}"
Ensure the generated plan focuses PRIMARILY on this request.
"""

# **Age**: {user_meta.get('age', 30)}
# **Gender**: {user_meta.get('gender', 'male')}
# **Weight**: {weight}kg
# **Fitness Level**: {fitness_level}
# **Medical Conditions**: {', '.join(conditions) if conditions else 'None'}

        prompt += f"""
## User Profile:
{user_meta}

## Requirements
**Target Intensity**: {target_intensity}
**Target Daily Calories Burn**: {target_calories} kcal
**Target Duration**: {target_duration} minutes per session
**Weekly Frequency**: {target_frequency} sessions per week

## Environment Context
**Weather**: {environment.get('weather', {})}
**Season**: {environment.get('time_context', {}).get('season', 'any')}

## Knowledge Graph Insights (Use these to optimize safety and effectiveness, but do not deviate from the USER REQUEST)
{kg_context}"""

        prompt += """
## Task
Generate a single exercise plan candidate. Return ONLY the JSON object, NO markdown code blocks, NO extra wrapper keys.
Each exercise MUST have: "name", "exercise_type", "duration_minutes", "intensity", "calories_burned".
Generate ONLY ONE session per day (single morning/afternoon/evening block).
"""
# Example format:
# {{
#   "id": 1,
#   "title": "Morning Cardio Plan",
#   "meal_timing": "after_breakfast",
#   "sessions": {{
#     "morning": {{
#       "time_of_day": "morning",
#       "exercises": [
#         {{
#           "name": "Brisk Walking",
#           "exercise_type": "cardio",
#           "duration_minutes": 30,
#           "intensity": "low",
#           "calories_burned": 135,
#           "equipment": [],
#           "target_muscles": ["legs", "cardio"],
#           "instructions": ["Walk at comfortable pace", "Maintain good posture"],
#           "reason": "Low-impact cardio suitable for beginners",
#           "safety_notes": ["Stay hydrated", "Warm up first"]
#         }}
#       ],
#       "total_duration_minutes": 30,
#       "total_calories_burned": 135,
#       "overall_intensity": "low"
#     }}
#   }},
#   "total_duration_minutes": 30,
#   "total_calories_burned": 135,
#   "reasoning": "This plan combines low-impact cardio with strength training",
#   "safety_notes": ["Consult physician before starting", "Listen to your body"]
# }}"""

        return prompt

    def _normalize_enum_values(self, data: Dict) -> Dict:
        """Normalize enum values to lowercase (LLM may return UPPERCASE)"""
        exercise_type_map = {
            "CARDIO": "cardio", "STRENGTH": "strength", "FLEXIBILITY": "flexibility",
            "BALANCE": "balance", "HIIT": "hiit"
        }
        intensity_map = {
            "LOW": "low", "MODERATE": "moderate", "HIGH": "high", "VERY_HIGH": "very_high"
        }
        time_map = {
            "MORNING": "morning", "AFTERNOON": "afternoon", "EVENING": "evening", "ANY": "any"
        }
        meal_timing_map = {
            "BEFORE_BREAKFAST": "before_breakfast", "AFTER_BREAKFAST": "after_breakfast",
            "BEFORE_LUNCH": "before_lunch", "AFTER_LUNCH": "after_lunch",
            "BEFORE_DINNER": "before_dinner", "AFTER_DINNER": "after_dinner"
        }

        def normalize_item(exercise: Dict) -> Dict:
            if exercise.get("exercise_type") in exercise_type_map:
                exercise["exercise_type"] = exercise_type_map[exercise["exercise_type"]]
            if exercise.get("intensity") in intensity_map:
                exercise["intensity"] = intensity_map[exercise["intensity"]]
            return exercise

        def normalize_session(session: Dict) -> Dict:
            if "overall_intensity" in session:
                session["overall_intensity"] = intensity_map.get(
                    session["overall_intensity"], session["overall_intensity"]
                )
            if "exercises" in session:
                session["exercises"] = [normalize_item(ex) for ex in session["exercises"]]
            return session

        if "sessions" in data:
            for key, session in data["sessions"].items():
                data["sessions"][key] = normalize_session(session)

        # Normalize meal_timing
        if data.get("meal_timing") in meal_timing_map:
            data["meal_timing"] = meal_timing_map[data["meal_timing"]]

        return data

    def _generate_single_candidate(
        self,
        user_prompt: str,
        candidate_id: int,
        fitness_level: str,
        weight: float,
        strategy: str = "balanced",
        temperature: float = 0.7
    ) -> Optional[ExercisePlan]:
        """Generate a single exercise plan candidate"""
        # Add strategy-specific guidance
        strategy_guidance = {
            "balanced": "Focus on mix of cardio, strength, and flexibility.",
            "variety": "Include diverse exercises to prevent boredom and plateaus.",
            "intensity_focus": "Emphasize appropriate intensity for fitness level."
        }

        full_prompt = user_prompt + f"\n\n### Strategy: {strategy.upper()}\n{strategy_guidance.get(strategy, 'Focus on balanced training.')}"

        # Call LLM
        try:
            EXERCISE_GENERATION_SYSTEM_PROMPT = GET_EXERCISE_GENERATION_SYSTEM_PROMPT()
            response = self._call_llm(
                system_prompt=EXERCISE_GENERATION_SYSTEM_PROMPT,
                user_prompt=full_prompt,
                temperature=temperature
            )

            # Handle empty response
            if not response or response == {}:
                print(f"[WARN] LLM returned empty response for candidate {candidate_id}")
                return None

            try:
                data = parse_json_response(response)
            except json.JSONDecodeError:
                print(f"[WARN] Invalid JSON from LLM: {response[:100]}...")
                return None

            # Handle different response formats
            if isinstance(data, list):
                plan_data = data[0] if data else {}
            elif isinstance(data, dict):
                plan_data = data
            else:
                return None

            # Normalize enum values (LLM may return UPPERCASE)
            plan_data = self._normalize_enum_values(plan_data)

            # Ensure ID is set
            if "id" not in plan_data:
                plan_data["id"] = candidate_id

            # Create ExercisePlan object
            return ExercisePlan(**plan_data)

        except Exception as e:
            print(f"Error generating exercise candidate {candidate_id}: {e}")
            return None


# ================= Convenience Functions =================

# def generate_exercise_candidates(
#     user_metadata: Dict[str, Any],
#     environment: Dict[str, Any] = {},
#     user_requirement: Dict[str, Any] = {},
#     num_base_plans: int = 3,
#     meal_timing: str = "",
#     user_preference: str = None,
#     use_vector: bool = False,
#     rag_topk: int = 3,
#     kg_context: str = None
# ) -> List[ExercisePlan]:
#     """
#     Convenience function to generate exercise candidates.

#     Args:
#         user_metadata: User physiological data
#         environment: Environmental context
#         user_requirement: User requirements (intensity, duration in minutes)
#         num_candidates: Number of candidates to generate
#         user_preference: User's string preference (e.g., "I want to focus on upper body exercises")
#         use_vector: Use vector search (GraphRAG) instead of keyword matching

#     Returns:
#         List of ExercisePlan objects
#     """
#     agent = ExerciseAgent()
#     input_data = {
#         "user_metadata": user_metadata,
#         "environment": environment,
#         "user_requirement": user_requirement,
#         # "num_candidates": num_candidates
#     }
#     return agent.generate(input_data, num_base_plans, meal_timing=meal_timing, user_preference=user_preference, use_vector=use_vector, rag_topk=rag_topk)


def generate_exercise_variants(
    user_metadata: Dict[str, Any],
    environment: Dict[str, Any] = {},
    user_requirement: Dict[str, Any] = {},
    num_base_plans: int = 3,
    num_var_plans: int = 3,
    min_scale: float = 0.7,
    max_scale: float = 1.3,
    meal_timing: str = "",
    user_preference: str = None,
    use_vector: bool = False,
    rag_topk: int = 3,
    kg_context: str = None,
    temperature: float = 0.7
) -> Dict[str, List[ExercisePlan]]:
    agent = ExerciseAgent()
    input_data = {
        "user_metadata": user_metadata,
        "environment": environment,
        "user_requirement": user_requirement,
        # "num_candidates": num_candidates
    }
    base_candidates, kg_context = agent.generate(
        input_data,
        num_base_plans,
        meal_timing=meal_timing,
        user_preference=user_preference,
        use_vector=use_vector,
        rag_topk=rag_topk,
        kg_context=kg_context,
        temperature=temperature
    )
    # Expand each candidate into variants
    parser = ExercisePlanParser(num_variants=num_var_plans, min_scale=min_scale, max_scale=max_scale)
    result = {}

    for base_plan in base_candidates:
        variants = parser.expand_plan(base_plan)
        result[base_plan.id] = variants

    return result, kg_context


if __name__ == "__main__":
    # Test the generator
    test_input = {
        "user_metadata": {
            "age": 35,
            "gender": "male",
            "height_cm": 175,
            "weight_kg": 70,
            "medical_conditions": ["diabetes"],
            "fitness_level": "intermediate"
        },
        "environment": {
            "weather": {"condition": "clear", "temperature_c": 25},
            "time_context": {"season": "summer"}
        },
        "user_requirement": {
            "goal": "weight_loss",
            "intensity": "moderate",
            "num_candidates": 2
        }
    }

    candidates = generate_exercise_candidates(**test_input)
    print(f"Generated {len(candidates)} exercise candidates")
    for c in candidates:
        print(f"- {c.title}: {c.total_calories_burned} kcal, {c.total_duration_minutes} min")

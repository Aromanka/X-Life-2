import json
import math
import os
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from agents.exercise.generator import generate_exercise_variants
from agents.safeguard.assessor import SafeguardAgent
from agents.safeguard.models import SafetyAssessment

import argparse


# Generate-Only Output

@dataclass
class ExerciseGenerateOnlyOutput:
    """Output from exercise generation only (without safety assessment)"""
    plans: List[Dict[str, Any]]      # Generated plans with variants
    generated_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling datetime serialization"""
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj
        return {
            "plans": convert_datetime(self.plans),
            "generated_at": self.generated_at
        }


# Pipeline Output

@dataclass
class ExercisePipelineOutput:
    """Output from the exercise pipeline"""
    all_plans: List[Dict[str, Any]]      # All expanded plans (for exer_plan.json)
    top_plans: List[Dict[str, Any]]       # Top K selected plans
    assessments: Dict[int, Dict[str, Any]]  # Safety assessments by plan ID
    generated_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling datetime serialization"""
        def convert_datetime(obj):
            """Recursively convert datetime to ISO string"""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj

        return {
            "all_plans": convert_datetime(self.all_plans),
            "top_plans": convert_datetime(self.top_plans),
            "assessments": convert_datetime(self.assessments),
            "generated_at": self.generated_at
        }
    

def exercise_generate(
        user_metadata: Dict[str, Any],
        environment: Dict[str, Any] = None,
        user_requirement: Dict[str, Any] = None,
        user_query: str = None,
        num_base_plans: int = 3,
        num_variants: int = 3,
        min_scale: float = 0.7,
        max_scale: float = 1.3,
        temperature: float = 0.7,
        top_p: float = 0.92,
        top_k: int = 50,
        top_k_selection: int = 3,
        output_path: str = "exer_plan.json",
        meal_timing: str = "",
        use_vector: bool = False,
        rag_topk: int = 3,
        verbose_on: bool = True
    ):
    all_plans_list = []
    kg_context = None
    if verbose_on and user_query:
        print(f"      User Query: \"{user_query}\"")
    for i in range(num_base_plans):
        variants_dict, kg_context = generate_exercise_variants(
            user_metadata=user_metadata,
            environment=environment,
            user_requirement=user_requirement,
            num_base_plans=1,
            num_var_plans=num_variants,
            min_scale=min_scale,
            max_scale=max_scale,
            meal_timing=meal_timing,
            user_preference=user_query,
            use_vector=use_vector,
            rag_topk=rag_topk,
            kg_context=kg_context,
            temperature=temperature
        )
        # Flatten variants into a single list
        for base_id, variants in variants_dict.items():
            variants_cnt = 0
            for variant_name, plan in variants.items():
                plan_dict = plan.model_dump()
                # A base plan id is reused by every scaled variant and may
                # also restart on each generation call.  Preserve it for
                # provenance, but assign a request-local id for assessment
                # joins and output lookup.
                plan_dict["_source_id"] = plan_dict.get("id")
                plan_dict["id"] = len(all_plans_list) + 1
                plan_dict["_variant"] = variant_name
                plan_dict["_base_id"] = base_id
                all_plans_list.append(plan_dict)
                variants_cnt += 1
            if verbose_on:
                print(f"      Base {i+1}/{num_base_plans}: (base_id={base_id}){variants_cnt} variants")
    return all_plans_list


# Exercise Pipeline

class ExercisePipeline:
    """
    Pipeline for generating and evaluating exercise plans.

    Generates multiple exercise candidates via LLM, expands them to
    intensity variants, assesses safety, and selects the best ones.
    """

    def __init__(self, world_model_scorer=None):
        self.safeguard = SafeguardAgent()
        self.world_model_scorer = world_model_scorer

    def generate(
        self,
        user_metadata: Dict[str, Any],
        environment: Dict[str, Any] = None,
        user_requirement: Dict[str, Any] = None,
        user_query: str = None,
        num_base_plans: int = 3,
        num_variants: int = 3,
        min_scale: float = 0.7,
        max_scale: float = 1.3,
        temperature: float = 0.7,
        top_p: float = 0.92,
        top_k: int = 50,
        top_k_selection: int = 3,
        output_path: str = "exer_plan.json",
        meal_timing: str = "",
        use_vector: bool = False,
        rag_topk: int = 3,
        world_model_scorer=None
    ) -> ExercisePipelineOutput:
        """
        Generate exercise options with safety assessment.

        Args:
            user_metadata: User physiological data
            environment: Environmental context
            user_requirement: User requirements (intensity, duration in minutes)
            user_query: Free-form user preference query (e.g., "I want to focus on upper body exercises")
            num_base_plans: Number of LLM-generated base plans
            num_variants: Number of intensity variants per base (Lite/Standard/Plus)
            min_scale: Minimum scale factor for variants
            max_scale: Maximum scale factor for variants
            temperature: LLM temperature (0.0-1.0)
            top_p: LLM top_p for nucleus sampling (0.0-1.0)
            top_k: Number of candidates to request from the generator
            output_path: Path to save all plans JSON
            meal_timing: Meal timing context
            use_vector: Use vector search (GraphRAG) instead of keyword matching
            rag_topk: Top-k similar entities for GraphRAG

        Returns:
            ExercisePipelineOutput with all plans, top plans, and assessments
        """

        # print("=" * 60)
        # print("EXERCISE PIPELINE")
        # print("=" * 60)
        # print(f"[INFO] LLM params: temp={temperature}, top_p={top_p}, top_k={top_k}")
        # print(f"[INFO] Selection: {num_base_plans} bases x {num_variants} variants -> top {top_k}")

        # Step 1: Generate exercise candidates with variants
        print(f"\n[1/4] Generating exercise candidates...")
        env = environment or {}
        req = user_requirement or {}
        all_plans_list = exercise_generate(
            user_metadata=user_metadata,
            environment=env,
            user_requirement=req,
            user_query=user_query,
            num_base_plans=num_base_plans,
            num_variants=num_variants,
            min_scale=min_scale,
            max_scale=max_scale,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            top_k_selection=top_k_selection,
            output_path=output_path,
            meal_timing=meal_timing,
            use_vector=use_vector,
            rag_topk=rag_topk,
        )

        if not all_plans_list:
            print("[WARN] No candidates generated!")
            return ExercisePipelineOutput(
                all_plans=[],
                top_plans=[],
                assessments={},
                generated_at=datetime.now().isoformat()
            )

        # Step 2: Assess each plan through safeguard
        print(f"\n[2/4] Assessing {len(all_plans_list)} plans through safeguard...")
        assessments: Dict[int, Dict[str, Any]] = {}
        for plan in all_plans_list:
            plan_id = plan.get("id", 0)
            assessment = self.safeguard.assess(
                plan=plan,
                plan_type="exercise",
                user_metadata=user_metadata,
                environment=env
            )
            assessments[plan_id] = assessment.model_dump()
            score = assessment.score
            is_safe = assessment.is_safe
            risk = assessment.risk_level.value
            variant = plan.get("_variant", "N/A")
            # print(f"      ID:{plan_id} {variant} | "
            #       f"Score:{score} | Risk:{risk} | Safe:{is_safe}")

        # Add assessment info to plans
        for plan in all_plans_list:
            plan_id = plan.get("id", 0)
            if plan_id in assessments:
                plan["_assessment"] = assessments[plan_id]

        # Step 3: Rank the safe candidates.
        print(f"\n[3/4] Selecting top {top_k_selection} safe plans...")

        # Safety is a gate: unsafe plans remain in ``all_plans`` for audit,
        # but cannot enter the final recommendation list.
        safe_plans = [
            plan for plan in all_plans_list
            if plan.get("_assessment", {}).get("is_safe") is True
        ]

        scorer = world_model_scorer or self.world_model_scorer
        if scorer is not None:
            for plan in safe_plans:
                try:
                    learned = scorer(plan, user_metadata, env)
                except TypeError:
                    learned = scorer(plan, user_metadata)
                if isinstance(learned, dict):
                    learned = learned.get("ranking_score", learned.get("score"))
                if learned is None or not math.isfinite(float(learned)):
                    raise ValueError("The world-model scorer must return a finite ranking score.")
                plan["_ranking_score"] = float(learned)

        # Sort safe plans by learned score when a world model is injected;
        # otherwise keep the deterministic safety-score fallback.
        sorted_plans = sorted(
            safe_plans,
            key=lambda p: p["_ranking_score"] if scorer is not None else p.get("_assessment", {}).get("score", 0),
            reverse=True
        )
        top_plans = sorted_plans[:top_k_selection]

        # for i, plan in enumerate(top_plans, 1):
        #     score = plan.get("_assessment", {}).get("score", 0)
        #     variant = plan.get("_variant", "N/A")
        #     print(f"      #{i} ID:{plan.get('id')} {variant} | Score:{score}")

        # Step 4: Save all plans to JSON
        print(f"\n[4/4] Saving all {len(all_plans_list)} plans to {output_path}...")
        output = ExercisePipelineOutput(
            all_plans=all_plans_list,
            top_plans=top_plans,
            assessments=assessments,
            generated_at=datetime.now().isoformat()
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"      Saved to {output_path}")

        return output

    def print_top_plans(self, output: ExercisePipelineOutput):
        """Print the top selected plans to terminal"""
        print(">>> TOP SELECTED EXERCISE PLANS")

        for i, plan in enumerate(output.top_plans, 1):
            assessment = plan.get("_assessment", {})
            print(f"\n#{i} Plan ID:{plan.get('id')} | {plan.get('_variant', 'N/A')}")
            print(f"   Title: {plan.get('title', 'N/A')}")
            print(f"   Meal Timing: {plan.get('meal_timing', 'N/A')}")
            print(f"   Safety Score: {assessment.get('score', 'N/A')}/100")
            print(f"   Risk Level: {assessment.get('risk_level', 'N/A')}")
            print(f"   Safe: {'Yes' if assessment.get('is_safe') else 'No'}")
            print(f"   Duration: {plan.get('total_duration_minutes', 'N/A')} min")
            print(f"   Calories Burned: {plan.get('total_calories_burned', 'N/A')} kcal")

            # Print sessions
            print("   Session:")
            sessions = plan.get("sessions", {})
            for time_key, session in sessions.items():
                print(f"     [{time_key.upper()}] {session.get('total_duration_minutes', 0)} min, "
                      f"{session.get('total_calories_burned', 0)} kcal, "
                      f"Intensity: {session.get('overall_intensity', 'N/A')}")
                # for ex in session.get("exercises", [])[:3]:
                for ex in session.get("exercises", []):
                    print(f"       - {ex.get('name', 'N/A')} ({ex.get('duration_minutes', 0)} min, "
                          f"{ex.get('intensity', 'N/A')})")

            if assessment.get("risk_factors"):
                print(f"   Risk Factors:")
                # for rf in assessment.get("risk_factors", [])[:3]:
                for rf in assessment.get("risk_factors", []):
                    print(f"     - {rf}")

            if assessment.get("recommendations"):
                print(f"   Recommendations:")
                # for rec in assessment.get("recommendations", [])[:2]:
                for rec in assessment.get("recommendations", []):
                    print(f"     - {rec}")

    def generate_only(
        self,
        user_metadata: Dict[str, Any],
        environment: Dict[str, Any] = None,
        user_requirement: Dict[str, Any] = None,
        user_query: str = None,
        num_base_plans: int = 3,
        num_variants: int = 3,
        min_scale: float = 0.7,
        max_scale: float = 1.3,
        temperature: float = 0.7,
        top_p: float = 0.92,
        top_k: int = 50,
        meal_timing: str = "",
        use_vector: bool = False,
        rag_topk: int = 3
    ) -> ExerciseGenerateOnlyOutput:
        """
        Generate exercise candidates only WITHOUT safety assessment.

        Args:
            user_metadata: User physiological data
            environment: Environmental context
            user_requirement: User requirements (intensity, duration)
            user_query: Free-form user preference query
            num_base_plans: Number of LLM-generated base plans
            num_variants: Number of intensity variants per base
            min_scale: Minimum scale factor for variants
            max_scale: Maximum scale factor for variants
            temperature: LLM temperature (0.0-1.0)
            top_p: LLM top_p for nucleus sampling
            meal_timing: Meal timing context
            use_vector: Use vector search (GraphRAG)
            rag_topk: Top-k similar entities for GraphRAG

        Returns:
            ExerciseGenerateOnlyOutput with generated plans only
        """
        print(f"\n[1/1] Generating exercise candidates (no assessment)...")
        env = environment or {}
        req = user_requirement or {}

        all_plans_list = exercise_generate(
            user_metadata=user_metadata,
            environment=env,
            user_requirement=req,
            user_query=user_query,
            num_base_plans=num_base_plans,
            num_variants=num_variants,
            min_scale=min_scale,
            max_scale=max_scale,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            meal_timing=meal_timing,
            use_vector=use_vector,
            rag_topk=rag_topk,
            verbose_on=True
        )

        return ExerciseGenerateOnlyOutput(
            plans=all_plans_list,
            generated_at=datetime.now().isoformat()
        )


# ================= Convenience Function =================

def run_exercise_pipeline(
    user_metadata: Dict[str, Any],
    environment: Dict[str, Any] = None,
    user_requirement: Dict[str, Any] = None,
    user_query: str = None,
    num_base_plans: int = 3,
    num_variants: int = 3,
    min_scale: float = 0.7,
    max_scale: float = 1.3,
    meal_timing: str = "",
    temperature: float = 0.7,
    top_p: float = 0.92,
    top_k: int = 50,
    top_k_selection: int = 3,
    output_path: str = "exer_plan.json",
    print_results: bool = True,
    use_vector: bool = False,
    rag_topk: int = 3
) -> ExercisePipelineOutput:
    """
    Run the exercise pipeline and optionally print results.

    Args:
        user_metadata: User physiological data
        environment: Environmental context
        user_requirement: User requirements (intensity, duration in minutes)
        user_query: Free-form user preference query (e.g., "I want to focus on upper body exercises")
        num_base_plans: Number of LLM-generated base plans
        num_variants: Number of intensity variants per base (Lite/Standard/Plus)
        min_scale: Minimum scale factor for variants
        max_scale: Maximum scale factor for variants
        meal_timing: Meal timing context
        temperature: LLM temperature (0.0-1.0)
        top_p: LLM top_p for nucleus sampling (0.0-1.0)
        top_k: Number of top plans to select by safety score
        output_path: Path to save all plans JSON
        print_results: Whether to print top plans to terminal
        use_vector: Use vector search (GraphRAG) instead of keyword matching
        rag_topk: Top-k similar entities for GraphRAG

    Returns:
        ExercisePipelineOutput object
    """
    pipeline = ExercisePipeline()
    output = pipeline.generate(
        user_metadata=user_metadata,
        environment=environment,
        user_requirement=user_requirement,
        user_query=user_query,
        num_base_plans=num_base_plans,
        num_variants=num_variants,
        min_scale=min_scale,
        max_scale=max_scale,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        top_k_selection=top_k_selection,
        output_path=output_path,
        meal_timing=meal_timing,
        use_vector=use_vector,
        rag_topk=rag_topk
    )

    if print_results:
        pipeline.print_top_plans(output)

    return output


# ================= CLI =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bn', type=int, default=3, help='base plan num')
    parser.add_argument('--vn', type=int, default=3, help='variation plan num')
    parser.add_argument('--topk', type=int, default=3, help='top k selection')
    parser.add_argument('--rag_topk', type=int, default=3, help='graph rag top_k similar entities')
    parser.add_argument('--use_vector', action='store_true', default=False, help='Use vector search (GraphRAG) instead of keyword matching')
    parser.add_argument('--min_scale', type=float, default=0.7, help='minimum scale factor for variants (default: 0.7)')
    parser.add_argument('--max_scale', type=float, default=1.3, help='maximum scale factor for variants (default: 1.3)')
    parser.add_argument('--meal_timing', type=str, default="before_breakfast", help='meal_timing must be one of: "before_breakfast", "after_breakfast", "before_lunch", "after_lunch", "before_dinner", "after_dinner".')
    parser.add_argument('--query', type=str, default="I want to focus on upper body exercises with moderate intensity", help='user query (free-form text for KG entity matching)')
    args = parser.parse_args()
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
            "intensity": "moderate",
            "duration": 30
        },
        "user_query": args.query,
        "use_vector": args.use_vector,
        "rag_topk": args.rag_topk,
        "num_base_plans": args.bn,
        "num_variants": args.vn,
        "min_scale": args.min_scale,
        "max_scale": args.max_scale,
        "temperature": 0.7,
        "top_k": 10,
        "top_k_selection": args.topk,
        "output_path": "exer_plan.json",
        "meal_timing": args.meal_timing
    }

    result = run_exercise_pipeline(**test_input)
    print(f"\nTotal plans generated: {len(result.all_plans)}")
    print(f"Top plans selected: {len(result.top_plans)}")

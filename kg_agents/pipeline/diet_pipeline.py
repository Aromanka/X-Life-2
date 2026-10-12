import json
import math
import os
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from agents.diet import generate_diet_candidates
from agents.diet.models import DietRecommendation
from agents.safeguard.assessor import SafeguardAgent
from agents.safeguard.models import SafetyAssessment

import argparse


@dataclass
class DietGenerateOnlyOutput:
    """Output from diet generation only (without safety assessment)"""
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


@dataclass
class DietPipelineOutput:
    """Output from the diet pipeline"""
    all_plans: List[Dict[str, Any]]      # All expanded plans (for plan.json)
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

class DietPipeline:
    """
    Pipeline for generating and evaluating lunch options.

    Generates multiple lunch candidates via LLM, expands them to
    portion variants, assesses safety, and selects the best ones.
    """

    def __init__(self, world_model_scorer=None):
        self.safeguard = SafeguardAgent()
        # Optional dependency-injected world-model scorer.  Loading the
        # research VLM is intentionally left to the application layer; when
        # supplied, this callable must return a learned ranking score for one
        # candidate and the same participant state.
        self.world_model_scorer = world_model_scorer

    def generate(
        self,
        user_metadata: Dict[str, Any],
        environment: Dict[str, Any] = None,
        user_requirement: Dict[str, Any] = None,
        user_query: str = None,
        num_base_plans: int = 3,
        num_variants: int = 3,
        min_scale: float = 0.5,
        max_scale: float = 1.5,
        meal_type: str = "lunch",
        temperature: float = 0.7,
        top_p: float = 0.92,
        top_k: int = 50,
        top_k_selection: int = 3,
        output_path: str = "plan.json",
        use_vector: bool = False,
        rag_topk: str = 3,
        world_model_scorer=None
    ) -> DietPipelineOutput:
        """
        Generate meal options with safety assessment.

        Args:
            user_metadata: User physiological data
            environment: Environmental context
            user_requirement: User goals (optional, can be empty)
            user_query: Free-form user preference query (e.g., "I want a tuna sandwich with vegetables")
            num_base_plans: Number of LLM-generated base plans
            num_variants: Number of portion variants per base (Lite/Standard/Plus)
            meal_type: Meal type to generate (breakfast/lunch/dinner/snacks)
            temperature: LLM temperature (0.0-1.0)
            top_p: LLM top_p for nucleus sampling (0.0-1.0)
            top_k: LLM top_k for top-k sampling
            top_k_selection: Number of safe plans to return after ranking
            output_path: Path to save all plans JSON
            use_vector: Use vector search (GraphRAG) instead of keyword matching
            rag_topk: top_k similar enetities for GraphRAG

        Returns:
            DietPipelineOutput with all plans, top plans, and assessments
        """
        env = environment or {}
        req = user_requirement or {}

        # print("=" * 60)
        # print(f"DIET PIPELINE ({meal_type.upper()})")
        # print("=" * 60)
        # print(f"[INFO] LLM params: temp={temperature}, top_p={top_p}, top_k={top_k}")
        # print(f"[INFO] Selection: {num_base_plans} bases x {num_variants} variants -> top {top_k_selection}")

        # Step 1: Generate meal candidates with variants
        print(f"\n[1/4] Generating {meal_type} candidates...")
        if user_query:
            print(f"      User Query: \"{user_query}\"")
        meal_candidates = []
        kg_context = None
        for i in range(num_base_plans):
            print("generate with kg_context=")
            print(kg_context)
            print("")
            candidates, kg_context = generate_diet_candidates(
                user_metadata=user_metadata,
                environment=env,
                user_requirement=req,
                num_variants=num_variants,
                min_scale=min_scale,
                max_scale=max_scale,
                meal_type=meal_type,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                user_preference=user_query,
                use_vector=use_vector,  # GraphRAG: use vector search instead of keyword matching
                rag_topk=rag_topk,
                kg_context=kg_context
            )
            meal_candidates.extend(candidates)
            print(f"      Base {i+1}/{num_base_plans}: {len(candidates)} variants")

        # Filter only lunch candidates (in case meal_type=None was passed)
        # meal_candidates = [c for c in candidates if c.meal_type == meal_type]
        # if not meal_candidates:
        #     meal_candidates = candidates  # Use all if already filtered

        print(f"      Found {len(meal_candidates)} {meal_type} candidates")

        if not meal_candidates:
            print("[WARN] No candidates generated!")
            return DietPipelineOutput(
                all_plans=[],
                top_plans=[],
                assessments={},
                generated_at=datetime.now().isoformat()
            )

        # Convert to dicts for assessment.  Generators restart their local
        # candidate counter on each base-plan call, so normalize identifiers
        # once per request before any assessment is stored or joined back.
        all_plans_dict = []
        for request_index, candidate in enumerate(meal_candidates, start=1):
            plan = candidate.model_dump()
            plan["_source_id"] = plan.get("id")
            plan["id"] = request_index
            all_plans_dict.append(plan)

        # Step 2: Assess each plan through safeguard
        print(f"\n[2/4] Assessing {len(all_plans_dict)} plans through safeguard...")
        assessments: Dict[int, Dict[str, Any]] = {}
        for plan in all_plans_dict:
            plan_id = plan.get("id", 0)
            assessment = self.safeguard.assess(
                plan=plan,
                plan_type="diet",
                user_metadata=user_metadata,
                environment=env
            )
            assessments[plan_id] = assessment.model_dump()
            score = assessment.score
            is_safe = assessment.is_safe
            risk = assessment.risk_level.value
            # print(f"      ID:{plan_id} {plan.get('variant', 'N/A')} | "
            #       f"Score:{score} | Risk:{risk} | Safe:{is_safe}")

        # Add assessment info to plans
        for plan in all_plans_dict:
            plan_id = plan.get("id", 0)
            if plan_id in assessments:
                plan["_assessment"] = assessments[plan_id]

        # Step 3: Rank the safe candidates.
        print(f"\n[3/4] Selecting top {top_k_selection} safe plans...")

        # Safety is a gate: only plans explicitly marked safe can become
        # recommendations.  Keep all plans in the output for auditability.
        safe_plans = [
            plan for plan in all_plans_dict
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
        # otherwise retain the deterministic safety-score fallback.
        sorted_plans = sorted(
            safe_plans,
            key=lambda p: p["_ranking_score"] if scorer is not None else p.get("_assessment", {}).get("score", 0),
            reverse=True
        )
        top_plans = sorted_plans[:top_k_selection]

        # for i, plan in enumerate(top_plans, 1):
        #     score = plan.get("_assessment", {}).get("score", 0)
        #     variant = plan.get("variant", "N/A")
        #     print(f"      #{i} ID:{plan.get('id')} {variant} | Score:{score}")

        # Step 4: Save all plans to JSON
        print(f"\n[4/4] Saving all {len(all_plans_dict)} plans to {output_path}...")
        output = DietPipelineOutput(
            all_plans=all_plans_dict,
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

    def print_top_plans(self, output: DietPipelineOutput):
        """Print the top selected plans to terminal"""
        print(">>> TOP SELECTED LUNCH PLANS")

        for i, plan in enumerate(output.top_plans, 1):
            assessment = plan.get("_assessment", {})
            print(f"\n#{i} Plan ID:{plan.get('id')} | {plan.get('variant', 'N/A')}")
            print(f"   Safety Score: {assessment.get('score', 'N/A')}/100")
            print(f"   Risk Level: {assessment.get('risk_level', 'N/A')}")
            print(f"   Safe: {'Yes' if assessment.get('is_safe') else 'No'}")
            print(f"   Calories: {plan.get('total_calories', 'N/A')} "
                  f"(Target: {plan.get('target_calories', 'N/A')}) "
                  f"Deviation: {plan.get('calories_deviation', 0)}%")

            print("   Food Items:")
            for item in plan.get("items", []):
                print(f"     - {item.get('food', 'N/A')}: "
                      f"{item.get('portion', 'N/A')} ({item.get('calories', 0)} kcal)")

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
        min_scale: float = 0.5,
        max_scale: float = 1.5,
        meal_type: str = "lunch",
        temperature: float = 0.7,
        top_p: float = 0.92,
        top_k: int = 50,
        use_vector: bool = False,
        rag_topk: str = 3
    ) -> DietGenerateOnlyOutput:
        """
        Generate meal candidates only WITHOUT safety assessment.

        Args:
            user_metadata: User physiological data
            environment: Environmental context
            user_requirement: User goals (optional, can be empty)
            user_query: Free-form user preference query
            num_base_plans: Number of LLM-generated base plans
            num_variants: Number of portion variants per base
            meal_type: Meal type to generate
            temperature: LLM temperature (0.0-1.0)
            top_p: LLM top_p for nucleus sampling
            top_k: LLM top_k for top-k sampling
            use_vector: Use vector search (GraphRAG)
            rag_topk: Top-k similar entities for GraphRAG

        Returns:
            DietGenerateOnlyOutput with generated plans only
        """
        env = environment or {}
        req = user_requirement or {}

        print(f"\n[1/1] Generating {meal_type} candidates (no assessment)...")
        if user_query:
            print(f"      User Query: \"{user_query}\"")

        meal_candidates = []
        kg_context = None
        for i in range(num_base_plans):
            candidates, kg_context = generate_diet_candidates(
                user_metadata=user_metadata,
                environment=env,
                user_requirement=req,
                num_variants=num_variants,
                min_scale=min_scale,
                max_scale=max_scale,
                meal_type=meal_type,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                user_preference=user_query,
                use_vector=use_vector,
                rag_topk=rag_topk,
                kg_context=kg_context
            )
            meal_candidates.extend(candidates)
            print(f"      Base {i+1}/{num_base_plans}: {len(candidates)} variants")

        print(f"      Found {len(meal_candidates)} {meal_type} candidates")

        # Convert to dicts
        all_plans_dict = [c.model_dump() for c in meal_candidates]

        return DietGenerateOnlyOutput(
            plans=all_plans_dict,
            generated_at=datetime.now().isoformat()
        )


def run_diet_pipeline(
    user_metadata: Dict[str, Any],
    environment: Dict[str, Any] = None,
    user_requirement: Dict[str, Any] = None,
    user_query: str = None,
    rag_topk: int = 3,
    num_base_plans: int = 3,
    num_variants: int = 3,
    min_scale: float = 0.5,
    max_scale: float = 1.5,
    meal_type: str = "lunch",
    temperature: float = 0.7,
    top_p: float = 0.92,
    top_k: int = 50,
    top_k_selection: int = 3,
    output_path: str = "plan.json",
    print_results: bool = True,
    use_vector: bool = False
) -> DietPipelineOutput:
    """
    Run the diet pipeline and optionally print results.

    Args:
        user_metadata: User physiological data
        environment: Environmental context
        user_requirement: User goals (optional, can be empty)
        user_query: Free-form user preference query (e.g., "I want a tuna sandwich with vegetables")
        num_base_plans: Number of LLM-generated base plans
        num_variants: Number of portion variants per base (Lite/Standard/Plus)
        meal_type: Meal type to generate (breakfast/lunch/dinner/snacks)
        temperature: LLM temperature (0.0-1.0)
        top_p: LLM top_p for nucleus sampling (0.0-1.0)
        top_k: LLM top_k for top-k sampling
        top_k_selection: Number of top plans to select by safety score
        output_path: Path to save all plans JSON
        print_results: Whether to print top plans to terminal
        use_vector: Use vector search (GraphRAG) instead of keyword matching

    Returns:
        DietPipelineOutput object
    """
    pipeline = DietPipeline()
    output = pipeline.generate(
        user_metadata=user_metadata,
        environment=environment,
        user_requirement=user_requirement,
        user_query=user_query,
        num_base_plans=num_base_plans,
        num_variants=num_variants,
        min_scale=min_scale,
        max_scale=max_scale,
        meal_type=meal_type,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        top_k_selection=top_k_selection,
        output_path=output_path,
        use_vector=use_vector,
        rag_topk=rag_topk
    )

    if print_results:
        pipeline.print_top_plans(output)

    return output


# ================= CLI =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diet Pipeline with KG Entity Matching')
    parser.add_argument('--bn', type=int, default=3, help='base plan num')
    parser.add_argument('--vn', type=int, default=3, help='var plan num')
    parser.add_argument('--topk', type=int, default=3, help='top k selection')
    parser.add_argument('--rag_topk', type=int, default=3, help='graph rag top_k similar entities')
    parser.add_argument('--use_vector', action='store_true', default=False, help='Use vector search (GraphRAG) instead of keyword matching')
    parser.add_argument('--min_scale', type=float, default=0.5, help='minimum scale factor for variants (default: 0.5)')
    parser.add_argument('--max_scale', type=float, default=1.5, help='maximum scale factor for variants (default: 1.5)')
    parser.add_argument('--query', type=str, default="I want a healthy tuna salad sandwich with fresh vegetables",
                       help='user query (free-form text for KG entity matching)')
    args = parser.parse_args()
    test_input = {
        "user_metadata": {
            "age": 35,
            "gender": "male",
            "height_cm": 175,
            "weight_kg": 70,
            "medical_conditions": ["diabetes"],
            "dietary_restrictions": ["low_sodium"],
            "fitness_level": "intermediate"
        },
        "environment": {
            "weather": {"condition": "clear", "temperature_c": 25},
            "time_context": {"season": "summer"}
        },
        "user_requirement": {},  # Empty, use user_query instead
        "user_query": args.query,  # Free-form query for KG entity matching
        "use_vector": args.use_vector,  # Use vector search (GraphRAG) instead of keyword matching
        "rag_topk": args.rag_topk,
        "num_base_plans": args.bn,
        "num_variants": args.vn,
        "min_scale": args.min_scale,
        "max_scale": args.max_scale,
        "meal_type": "lunch",
        "temperature": 0.9,
        "top_k": args.topk,
        "output_path": "plan.json"
    }

    result = run_diet_pipeline(**test_input)
    print(f"\nTotal plans generated: {len(result.all_plans)}")
    print(f"Top plans selected: {len(result.top_plans)}")

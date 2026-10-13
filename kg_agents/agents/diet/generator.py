import json
import random
import re
from typing import List, Dict, Any, Optional
from agents.base import BaseAgent, DietAgentMixin
from agents.diet.models import (
    FoodItem,
    DietRecommendation, DietAgentInput,
    BaseFoodItem
)
from agents.diet.parser_var import DietPlanParser
from core.llm.utils import parse_json_response
from agents.diet.config import *
from core.prompts import (
    available_strategies, available_cuisines, GET_DIET_GENERATION_SYSTEM_PROMPT,
    build_diet_prompt
)



def _to_food_item(item_dict: Dict[str, Any]) -> FoodItem:
    """Transform parser output to FoodItem format for DietRecommendation"""
    return FoodItem(
        food=item_dict.get("food_name", ""),
        portion=f"{item_dict.get('portion_number', '')}{item_dict.get('portion_unit', '')}",
        calories=int(round(item_dict.get("total_calories", 0))),
    )


def build_constraint_prompt(protein: str, carb: str, veg: str, excluded: List[str] = None) -> str:
    # prompt = "\n## Mandatory Ingredients (YOU MUST USE THESE)\n"
    # prompt += f"- Main Protein: {protein}\n"
    # prompt += f"- Carb Source: {carb}\n"
    # prompt += f"- Vegetable: {veg}\n"
    prompt = ""
    if excluded:
        prompt += f"\n## Excluded Ingredients (DO NOT USE)\n"
        prompt += f"- {', '.join(excluded)}\n"

    return prompt


# ================= Diet Agent =================

class DietAgent(BaseAgent, DietAgentMixin):
    """Agent for generating diet recommendation candidates"""

    def __init__(self, num_variants: int = 3, min_scale: float = 0.5, max_scale: float = 1.5):
        super().__init__()
        self.parser = DietPlanParser(num_variants=num_variants, min_scale=min_scale, max_scale=max_scale)
        self.num_variants = num_variants
        self.min_scale = min_scale
        self.max_scale = max_scale

    def get_agent_name(self) -> str:
        return "diet"

    def get_input_type(self):
        return DietAgentInput

    def get_output_type(self):
        return DietRecommendation

    def generate(
        self,
        input_data: Dict[str, Any],
        num_variants: int = 3,
        min_scale: float = 0.5,
        max_scale: float = 1.5,
        meal_type: str = None,
        temperature: float = 0.7,
        top_p: float = 0.92,
        top_k: int = 50,
        user_preference: str = None,
        use_vector: bool = True,  # GraphRAG: use vector search instead of keyword matching
        rag_topk: int = 3,
        kg_context: str = None
    ) -> List[DietRecommendation]:
        # Reinitialize parser if variant configuration changed
        if (num_variants != self.num_variants or
            min_scale != self.min_scale or
            max_scale != self.max_scale):
            self.parser = DietPlanParser(num_variants=num_variants, min_scale=min_scale, max_scale=max_scale)
            self.num_variants = num_variants
            self.min_scale = min_scale
            self.max_scale = max_scale
        # KG Format Version
        KG_FORMAT_VER = 3

        # Parse input
        input_obj = DietAgentInput(**input_data)

        user_meta = input_obj.user_metadata
        env = input_obj.environment
        requirement = input_obj.user_requirement

        target_calories = self.calculate_target_calories(
            age=user_meta.get("age", 30),
            gender=user_meta.get("gender", "male"),
            height_cm=user_meta.get("height_cm", 170),
            weight_kg=user_meta.get("weight_kg", 70),
            goal=requirement.get("goal", "maintenance"),
            activity_factor=self._get_activity_factor(user_meta.get("fitness_level", "beginner"))
        )

        if kg_context is None:
            # Get KG context
            kg_context = ""
            conditions = user_meta.get("medical_conditions", [])

            # Query condition-based KG context
            if conditions:
                dietary_knowledge = self.query_dietary_knowledge(
                    conditions, user_meta.get("dietary_restrictions", [])
                )
                kg_context = self._format_kg_context(dietary_knowledge)

            # Query entity-based KG context when user_preference is provided
            if user_preference:
                entity_knowledge = self.query_dietary_by_entity(
                    user_preference,
                    use_vector_search=use_vector,
                    rag_topk=rag_topk,
                    kg_format_ver=KG_FORMAT_VER
                )
                entity_context = self._format_dietary_entity_kg_context(entity_knowledge, kg_format_ver=KG_FORMAT_VER)
                kg_context += entity_context
        else:
            pass

        # Define meal types to generate
        if meal_type:
            meal_types = [meal_type]
        else:
            meal_types = ["breakfast", "lunch", "dinner", "snacks"]

        # Get variant names from parser configuration
        variant_names = [name for name, _ in self.parser.variant_configs]
        
        used_strategies = set()
        used_combinations = set()

        # Collect base plans for each meal type
        meal_base_plans: Dict[str, Dict[str, Any]] = {}

        for mt in meal_types:
            # Select strategy and cuisine - DISABLE random constraints when user_preference exists
            # When user has a specific request, let LLM decide based on user intent
            if user_preference:
                strategy = "User-Directed"  # Tell Prompt this is a user-directed task
                cuisine = "As Requested"   # Let LLM infer from query
                excluded = []
            else:
                remaining_strategies = [s for s in available_strategies if s not in used_strategies]
                if not remaining_strategies:
                    remaining_strategies = available_strategies

                strategy = random.choice(remaining_strategies)
                used_strategies.add(strategy)

                cuisine = random.choice(available_cuisines)

                excluded = []
                if random.random() > 0.5:
                    excluded = random.sample(COMMON_BORING_FOODS, k=random.randint(1, 2))
                    
            base_items = self._generate_base_plan(
                user_meta=user_meta,
                environment=env,
                requirement=requirement,
                target_calories=target_calories,
                meal_type=mt,
                kg_context=kg_context,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                strategy=strategy,
                cuisine=cuisine,
                # constraint_prompt=constraint_prompt,
                constraint_prompt="",
                user_preference=user_preference
            )
            if base_items:
                meal_base_plans[mt] = {
                    "items": base_items,
                    "strategy": strategy,
                    "cuisine": cuisine,
                    "excluded": excluded
                }

        if not meal_base_plans:
            print("[WARN] No base plans generated for any meal type")
            return []

        # Expand each meal to variants
        expanded_meals: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for meal_type, plan_info in meal_base_plans.items():
            expanded_meals[meal_type] = self.parser.expand_plan(plan_info["items"], variant_names)

        candidates = []
        candidate_id = 1

        # Calorie targets per meal type
        meal_targets = {
            "breakfast": int(target_calories * 0.25),
            "lunch": int(target_calories * 0.35),
            "dinner": int(target_calories * 0.30),
            "snacks": int(target_calories * 0.10)
        }

        for meal_type in meal_types:
            if meal_type not in expanded_meals:
                continue

            # Get strategy and cuisine for this meal
            plan_info = meal_base_plans.get(meal_type, {})
            strategy = plan_info.get("strategy", "balanced")
            cuisine = plan_info.get("cuisine", "General")

            meal_variants = expanded_meals[meal_type]
            target = meal_targets.get(meal_type, int(target_calories * 0.25))

            for variant_name in variant_names:
                meal_items = meal_variants.get(variant_name, [])
                if not meal_items:
                    continue

                # Transform to FoodItem format
                food_items = [_to_food_item(item) for item in meal_items]
                total_cal = sum(item.calories for item in food_items)

                # Calculate deviation
                deviation = round(((total_cal - target) / target) * 100, 1)

                # Build safety notes
                safety_notes = [f"Meal: {meal_type}", f"Variant: {variant_name}"]
                safety_notes.append(f"Style: {cuisine}, Strategy: {strategy}")
                excluded = plan_info.get("excluded", [])
                if excluded:
                    safety_notes.append(f"Excluded: {', '.join(excluded)}")
                if abs(deviation) > 10:
                    safety_notes.append(f"Calorie deviation: {deviation}%")

                candidate = DietRecommendation(
                    id=candidate_id,
                    meal_type=meal_type,
                    variant=variant_name,
                    items=food_items,
                    total_calories=int(total_cal),
                    target_calories=target,
                    calories_deviation=deviation,
                    safety_notes=safety_notes
                )
                candidates.append(candidate)
                candidate_id += 1

        # Sort by deviation
        candidates.sort(key=lambda x: (x.meal_type, abs(x.calories_deviation)))

        return candidates, kg_context

    def _generate_base_plan(
        self,
        user_meta: Dict[str, Any],
        environment: Dict[str, Any],
        requirement: Dict[str, Any],
        target_calories: int,
        meal_type: str,
        kg_context: str = "",
        temperature: float = 0.85,
        top_p: float = 0.92,
        top_k: int = 50,
        strategy: str = "balanced",
        cuisine: str = "General",
        constraint_prompt: str = "",
        user_preference: str = None
    ) -> Optional[List[BaseFoodItem]]:
        """Generate base food items for a single meal type with diversity injection"""

        strategy_guidance = {
            "balanced": "Focus on balanced nutrition across all macros.",
            "protein_focus": "Emphasize high-protein foods for muscle maintenance.",
            "variety": "Include diverse food types and colors.",
            "low_carb": "Reduce carbohydrate intake slightly, focus on quality fats and proteins.",
            "fiber_rich": "Prioritize high-fiber vegetables and whole grains."
        }

        # user_prompt = self._build_diet_prompt(
        user_prompt = build_diet_prompt(
            user_meta=user_meta,
            environment=environment,
            requirement=requirement,
            target_calories=target_calories,
            meal_type=meal_type,
            kg_context=kg_context,
            user_preference=user_preference
        )

        full_prompt = user_prompt
        # full_prompt = user_prompt + f"\n\n### Optimization Strategy: {strategy.upper()}\n{strategy_guidance.get(strategy, '')}"
        # full_prompt += f"\n\n### Culinary Style: {cuisine}\nPLEASE strictly follow this style. Use ingredients and cooking methods typical for {cuisine} cuisine."
        # full_prompt += constraint_prompt

        DIET_GENERATION_SYSTEM_PROMPT = GET_DIET_GENERATION_SYSTEM_PROMPT()
        response = self._call_llm(
            system_prompt=DIET_GENERATION_SYSTEM_PROMPT,
            user_prompt=full_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )

        if not response or response == {}:
            print(f"[WARN] LLM returned empty for {meal_type}")
            return None

        try:
            data = parse_json_response(response)
        except json.JSONDecodeError as e:
            print(f"[WARN] Invalid JSON for {meal_type}: {e}")
            return None

        # Parse to BaseFoodItem list
        if isinstance(data, list):
            items = []
            for i, item_data in enumerate(data):
                try:
                    item = BaseFoodItem(**item_data)
                    items.append(item)
                except Exception as e:
                    print(f"[WARN] Failed to parse item {i}: {e}")
            return items if items else None
        else:
            print(f"[WARN] Expected list, got {type(data)}")
            return None

    def _get_activity_factor(self, fitness_level: str) -> float:
        """Get activity factor from fitness level"""
        factors = {
            "sedentary": 1.2,
            "beginner": 1.375,
            "intermediate": 1.55,
            "advanced": 1.725
        }
        return factors.get(fitness_level, 1.2)

    def _format_kg_context(self, knowledge: List) -> str:
        """Format KG knowledge for prompt"""
        if not knowledge:
            return ""

        parts = []

        # set maximum input lengths = 20
        maximum_inputs = 20
        if len(knowledge)>maximum_inputs:
            random.shuffle(knowledge)
            knowledge = knowledge[:maximum_inputs]
        
        for item in knowledge:
            entity_name = item.get('entity', "name")
            rel = item.get('rel', "relation")
            tail = item.get('tail', "name")
            condition = item.get('condition', "condition")

            # condition_words = "under condition: {}".format(condition)
            # condition_words = "under condition: {}".format(condition)
            # condition_words = "regarding {} issues".format(condition)
            # part = "{} {} {} {}".format(entity_name, rel, tail, condition_words)
            part = "<{}, {}, {}> regarding {} issues".format(entity_name, rel, tail, condition)
            parts.append(part)

        return "#### Profile based KG Guidelines:\n" + "\n".join(parts) + "\n"


# ================= Convenience Functions =================

def generate_diet_candidates(
    user_metadata: Dict[str, Any],
    environment: Dict[str, Any] = {},
    user_requirement: Dict[str, Any] = {},
    num_variants: int = 3,
    min_scale: float = 0.5,
    max_scale: float = 1.5,
    meal_type: str = None,
    temperature: float = 0.7,
    top_p: float = 0.92,
    top_k: int = 50,
    user_preference: str = None,
    use_vector: bool = False,
    rag_topk: str = 3,
    kg_context: str = None
) -> List[DietRecommendation]:
    agent = DietAgent(num_variants=num_variants, min_scale=min_scale, max_scale=max_scale)
    input_data = {
        "user_metadata": user_metadata,
        "environment": environment,
        "user_requirement": user_requirement,
    }
    return agent.generate(
        input_data, num_variants, min_scale, max_scale,
        meal_type, temperature, top_p, top_k, user_preference, use_vector, rag_topk,
        kg_context=kg_context
    )


if __name__ == "__main__":
    # Test the generator
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
        "user_requirement": {
            "goal": "weight_loss"
        },
        "num_variants": 3
    }

    candidates = generate_diet_candidates(**test_input)
    # DEBUG print
    # print(f"Generated {len(candidates)} diet candidates")
    # for c in candidates:
    #     print(f"\n[ID:{c.id}] {c.meal_type.upper()} - {c.variant}")
    #     print(f"  Target: {c.target_calories} kcal | Actual: {c.total_calories} kcal | Deviation: {c.calories_deviation}%")
    #     for item in c.items:
    #         print(f"  - {item.food}: {item.portion} ({item.calories} kcal)")


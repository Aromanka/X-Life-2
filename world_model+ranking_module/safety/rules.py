"""Deterministic policy migrated from the original SafeguardAgent, without LLM imports."""
import math
import json
from typing import Dict, Any, Optional, List
from .rule_models import RiskLevel, RiskFactor, SafetyCheck
from .policy import get_DIET_SAFETY_RULES, get_EXERCISE_SAFETY_RULES, get_CONDITION_RESTRICTIONS
DIET_SAFETY_RULES=get_DIET_SAFETY_RULES(RiskLevel)
EXERCISE_SAFETY_RULES=get_EXERCISE_SAFETY_RULES(RiskLevel)
CONDITION_RESTRICTIONS=get_CONDITION_RESTRICTIONS()

class SafeguardRules:
    def _check_diet_safety(
        self,
        plan: Dict[str, Any],
        user_metadata: Dict[str, Any]
    ) -> tuple:
        checks = []
        risk_factors = []

        def _as_number(value: Any) -> Optional[float]:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        # DietRecommendation.total_calories represents one meal.  Daily
        # calorie limits must only be applied when a daily total is explicitly
        # supplied by the caller; otherwise a normal 500 kcal lunch would be
        # incorrectly rejected as an unsafe all-day diet.
        total_calories = _as_number(plan.get("total_calories"))
        daily_calories = next(
            (
                _as_number(plan.get(key))
                for key in (
                    "daily_total_calories",
                    "total_daily_calories",
                    "daily_calories",
                )
                if plan.get(key) is not None
            ),
            None,
        )
        is_daily_plan = daily_calories is not None or bool(plan.get("is_daily_plan"))
        if is_daily_plan and daily_calories is None:
            daily_calories = total_calories

        if is_daily_plan and daily_calories is not None:
            if daily_calories < DIET_SAFETY_RULES["min_calories"]["value"]:
                checks.append(SafetyCheck(
                    check_name="min_calories",
                    passed=False,
                    message=DIET_SAFETY_RULES["min_calories"]["message"],
                    severity=RiskLevel.HIGH
                ))
                risk_factors.append(RiskFactor(
                    factor="extremely_low_calories",
                    category="nutritional",
                    severity=RiskLevel.HIGH,
                    description=f"Daily calories {daily_calories:g} is dangerously low",
                    recommendation="Consult a dietitian for safe calorie targets"
                ))
            elif daily_calories > DIET_SAFETY_RULES["max_calories"]["value"]:
                checks.append(SafetyCheck(
                    check_name="max_calories",
                    passed=False,
                    message=DIET_SAFETY_RULES["max_calories"]["message"],
                    severity=RiskLevel.MODERATE
                ))
            else:
                checks.append(SafetyCheck(
                    check_name="calories_range",
                    passed=True,
                    message="Daily calorie intake within acceptable range"
                ))
        elif total_calories is not None:
            max_meal = DIET_SAFETY_RULES["single_meal_calories"]["value"]
            checks.append(SafetyCheck(
                check_name="single_meal_calories",
                passed=total_calories <= max_meal,
                message=(
                    "Meal calories within acceptable range"
                    if total_calories <= max_meal
                    else f"Meal calories {total_calories:g} exceeds {max_meal} kcal"
                ),
                severity=None if total_calories <= max_meal else RiskLevel.LOW
            ))
            if total_calories > max_meal:
                risk_factors.append(RiskFactor(
                    factor="high_single_meal_calories",
                    category="nutritional",
                    severity=RiskLevel.LOW,
                    description=f"Meal calories {total_calories:g} is high",
                    recommendation="Distribute calories across meals"
                ))

        macros = plan.get("macro_nutrients", {})
        if macros:
            if isinstance(macros, dict):
                protein_ratio = macros.get("protein_ratio", 0)
                fat_ratio = macros.get("fat_ratio", 0)
            else:
                protein_ratio = getattr(macros, "protein_ratio", 0)
                fat_ratio = getattr(macros, "fat_ratio", 0)

            if protein_ratio < DIET_SAFETY_RULES["min_protein_ratio"]["value"]:
                risk_factors.append(RiskFactor(
                    factor="low_protein",
                    category="nutritional",
                    severity=RiskLevel.MODERATE,
                    description=f"Protein ratio {protein_ratio*100:.1f}% is below recommended minimum",
                    recommendation="Include more protein-rich foods"
                ))

            if fat_ratio > DIET_SAFETY_RULES["max_fat_ratio"]["value"]:
                risk_factors.append(RiskFactor(
                    factor="high_fat",
                    category="nutritional",
                    severity=RiskLevel.MODERATE,
                    description=f"Fat ratio {fat_ratio*100:.1f}% exceeds recommended maximum",
                    recommendation="Reduce high-fat foods"
                ))

        meal_plan = plan.get("meal_plan", {})
        if meal_plan:
            for meal_type, items in meal_plan.items():
                if isinstance(items, list):
                    meal_calories = sum(
                        _as_number(item.get("calories")) or 0
                        for item in items if isinstance(item, dict)
                    )
                elif isinstance(items, dict):
                    # Might be a MealPlanItem object
                    meal_calories = _as_number(items.get("total_calories")) or 0
                else:
                    meal_calories = 0

                # ``total_calories`` above already covers the current
                # DietRecommendation.  This loop is for explicit multi-meal
                # plans only, avoiding duplicate findings for one meal.
                if (
                    meal_calories > DIET_SAFETY_RULES["single_meal_calories"]["value"]
                    and not (
                        isinstance(plan.get("items"), list)
                        and meal_type == plan.get("meal_type")
                    )
                ):
                    checks.append(SafetyCheck(
                        check_name="single_meal_calories",
                        passed=False,
                        message=f"{meal_type} calorie {meal_calories} is high",
                        severity=RiskLevel.LOW
                    ))

        return checks, risk_factors

    def _check_exercise_safety(
        self,
        plan: Dict[str, Any],
        user_metadata: Dict[str, Any]
    ) -> tuple:
        checks = []
        risk_factors = []

        # ExercisePlan fields
        fitness_level = user_metadata.get("fitness_level", "beginner")
        total_duration = plan.get("total_duration_minutes", 0)
        sessions = plan.get("sessions", {})
        weekly_freq = plan.get("weekly_frequency", 3)

        # Duration limits based on fitness level
        max_duration_map = {
            "beginner": EXERCISE_SAFETY_RULES["max_daily_duration_beginner"]["value"],
            "intermediate": EXERCISE_SAFETY_RULES["max_daily_duration_intermediate"]["value"],
            "advanced": EXERCISE_SAFETY_RULES["max_daily_duration_advanced"]["value"]
        }
        max_duration = max_duration_map.get(fitness_level, 60)

        if total_duration > max_duration:
            severity = RiskLevel.MODERATE if fitness_level == "advanced" else RiskLevel.HIGH
            checks.append(SafetyCheck(
                check_name="daily_duration",
                passed=False,
                message=f"Duration {total_duration}min exceeds {fitness_level} limit ({max_duration}min)",
                severity=severity
            ))
            risk_factors.append(RiskFactor(
                factor="excessive_duration",
                category="exercise",
                severity=severity,
                description=f"Total exercise time {total_duration}min is excessive for {fitness_level}",
                recommendation=f"Reduce daily duration to {max_duration}min or less"
            ))
        else:
            checks.append(SafetyCheck(
                check_name="daily_duration",
                passed=True,
                message=f"Duration {total_duration}min is appropriate"
            ))

        # Rest day check
        if weekly_freq > EXERCISE_SAFETY_RULES["max_weekly_sessions"]["value"]:
            checks.append(SafetyCheck(
                check_name="rest_days",
                passed=False,
                message="Exercise every day without rest",
                severity=RiskLevel.MODERATE
            ))
            risk_factors.append(RiskFactor(
                factor="no_rest_days",
                category="exercise",
                severity=RiskLevel.MODERATE,
                description="No rest days scheduled in weekly plan",
                recommendation="Include at least 1-2 rest days per week"
            ))

        # HIIT frequency check - handle both string and enum formats
        def get_exercise_type(ex):
            """Extract exercise type, handling both string and enum formats"""
            et = ex.get("exercise_type", "")
            if isinstance(et, str):
                return et.lower()
            # Handle enum
            return str(et).lower() if hasattr(et, 'value') else str(et).lower()

        has_hiit = False
        for session_key, session in sessions.items():
            # session can be dict or ExerciseSession object
            exercises = session.get("exercises", []) if isinstance(session, dict) else []
            if not exercises and hasattr(session, 'exercises'):
                exercises = session.exercises

            for ex in exercises:
                if isinstance(ex, dict):
                    ex_type = get_exercise_type(ex)
                else:
                    # Object format
                    ex_type = str(ex.exercise_type).lower() if hasattr(ex, 'exercise_type') else ""
                if ex_type == "hiit":
                    has_hiit = True
                    break
            if has_hiit:
                break

        if has_hiit and weekly_freq > 3:
            checks.append(SafetyCheck(
                check_name="hiit_frequency",
                passed=False,
                message="HIIT sessions too frequent without adequate recovery",
                severity=RiskLevel.HIGH
            ))
            risk_factors.append(RiskFactor(
                factor="hiit_frequency",
                category="exercise",
                severity=RiskLevel.HIGH,
                description="HIIT sessions too frequent without adequate recovery",
                recommendation="Limit HIIT to 2-3 times per week with 48h rest"
            ))

        # High intensity check for beginners/intermediate
        if fitness_level in ["beginner", "intermediate"]:
            has_high_intensity = False
            for session_key, session in sessions.items():
                exercises = session.get("exercises", []) if isinstance(session, dict) else []
                if not exercises and hasattr(session, 'exercises'):
                    exercises = session.exercises

                for ex in exercises:
                    if isinstance(ex, dict):
                        intensity = str(ex.get("intensity", "")).lower()
                    else:
                        intensity = str(ex.intensity).lower() if hasattr(ex, 'intensity') else ""
                    if intensity == "high" or intensity == "very_high":
                        has_high_intensity = True
                        break
                if has_high_intensity:
                    break

            if has_high_intensity:
                checks.append(SafetyCheck(
                    check_name="intensity_level",
                    passed=False,
                    message=f"High intensity exercise may not be suitable for {fitness_level}",
                    severity=RiskLevel.MODERATE
                ))

        return checks, risk_factors

    def _extract_plan_content_text(self, plan: Dict[str, Any], plan_type: str) -> str:
        """Extract readable text from plan for condition checking"""
        content_parts = []

        if plan_type == "diet":
            # Extract food items
            direct_items = plan.get("items", [])
            if isinstance(direct_items, list):
                for item in direct_items:
                    if isinstance(item, dict):
                        food_name = item.get("food") or item.get("food_name", "")
                        if food_name:
                            content_parts.append(str(food_name))
                    elif hasattr(item, "food"):
                        content_parts.append(str(item.food))

            meal_plan = plan.get("meal_plan", {})
            if isinstance(meal_plan, dict):
                for meal_type, items in meal_plan.items():
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                food_name = item.get("food") or item.get("food_name", "")
                                if food_name:
                                    content_parts.append(food_name)
                            elif hasattr(item, 'food'):
                                content_parts.append(item.food)
            # Add macro info
            macros = plan.get("macro_nutrients", {})
            if isinstance(macros, dict):
                content_parts.extend([str(v) for v in macros.values()])

        elif plan_type == "exercise":
            # Extract exercise names and types
            sessions = plan.get("sessions", {})
            if isinstance(sessions, dict):
                for session_key, session in sessions.items():
                    if isinstance(session, dict):
                        intensity = session.get('overall_intensity')
                        if intensity:
                            content_parts.append(f"intensity:{getattr(intensity, 'value', intensity)}")
                        exercises = session.get("exercises", [])
                        for ex in exercises:
                            if isinstance(ex, dict):
                                ex_name = ex.get("name", "")
                                ex_type = ex.get("exercise_type", "")
                                if ex_name:
                                    content_parts.append(ex_name)
                                if ex_type:
                                    content_parts.append(str(ex_type))
                                if ex.get("intensity"):
                                    intensity = ex["intensity"]
                                    content_parts.append(f"intensity:{getattr(intensity, 'value', intensity)}")
                            elif hasattr(ex, 'name'):
                                content_parts.append(ex.name)
                                if hasattr(ex, 'exercise_type'):
                                    content_parts.append(str(ex.exercise_type))

        return " ".join(content_parts).lower()

    def _check_condition_restrictions(
        self,
        plan: Dict[str, Any],
        plan_type: str,
        user_metadata: Dict[str, Any]
    ) -> tuple:
        checks = []
        risk_factors = []

        conditions = user_metadata.get("medical_conditions", [])
        if isinstance(conditions, str):
            conditions = [conditions]

        plan_content = self._extract_plan_content_text(plan, plan_type)

        for condition in conditions:
            condition_lower = str(condition).lower()
            matched_condition = None
            for known_condition in CONDITION_RESTRICTIONS:
                if condition_lower == known_condition or condition_lower.replace("_", "") == known_condition.replace("_", ""):
                    matched_condition = known_condition
                    break

            if matched_condition:
                restrictions = CONDITION_RESTRICTIONS[matched_condition]

                # Check diet restrictions
                if plan_type == "diet" and "diet" in restrictions:
                    diet_rules = restrictions["diet"]

                    for rule_key, rule_desc in diet_rules.items():
                        if "avoid" in rule_key or "max" in rule_key:
                            # Configuration uses names such as
                            # ``avoid_high_sugar`` while the keyword table is
                            # keyed by ``high_sugar``.  Normalize both forms.
                            forbidden_keywords = {
                                "high_sugar": ["sugar", "candy", "cake", "ice cream", "sweet"],
                                "high_sodium": ["salt", "soy sauce", "pickled", "processed"],
                                "high_cholesterol": ["egg yolk", "liver", "organ meat"],
                                "caffeine": ["coffee", "tea", "caffeine"],
                            }
                            keyword_key = rule_key.removeprefix("avoid_")
                            keywords = forbidden_keywords.get(keyword_key, [])
                            for keyword in keywords:
                                if keyword in plan_content:
                                    checks.append(SafetyCheck(
                                        check_name=f"{matched_condition}_{rule_key}",
                                        passed=False,
                                        message=f"Plan contains {keyword} which should be avoided for {condition}",
                                        severity=RiskLevel.HIGH
                                    ))
                                    risk_factors.append(RiskFactor(
                                        factor=f"{matched_condition}_{rule_key}",
                                        category="medical",
                                        severity=RiskLevel.HIGH,
                                        description=f"Plan contains {keyword} which should be avoided for {condition}",
                                        recommendation=f"Remove {rule_desc} for {condition} management"
                                    ))

                            # ``max_sodium`` is a numeric daily limit.  Read it
                            # only when the caller provides an explicit sodium
                            # value; otherwise leave the rule auditable but do
                            # not invent nutrition data for a single meal.
                            if rule_key == "max_sodium":
                                sodium = next(
                                    (
                                        plan.get(key)
                                        for key in (
                                            "daily_sodium_mg",
                                            "total_sodium_mg",
                                            "sodium_mg",
                                            "sodium",
                                        )
                                        if plan.get(key) is not None
                                    ),
                                    None,
                                )
                                try:
                                    sodium = float(sodium)
                                except (TypeError, ValueError):
                                    sodium = None
                                if sodium is not None and sodium > float(rule_desc):
                                    checks.append(SafetyCheck(
                                        check_name=f"{matched_condition}_{rule_key}",
                                        passed=False,
                                        message=f"Sodium {sodium:g}mg exceeds {rule_desc}mg",
                                        severity=RiskLevel.HIGH
                                    ))
                                    risk_factors.append(RiskFactor(
                                        factor=f"{matched_condition}_{rule_key}",
                                        category="medical",
                                        severity=RiskLevel.HIGH,
                                        description=f"Sodium {sodium:g}mg exceeds the recommended limit",
                                        recommendation="Reduce sodium-rich foods"
                                    ))

                # Check exercise restrictions
                if plan_type == "exercise" and "exercise" in restrictions:
                    ex_rules = restrictions["exercise"]

                    for rule_key, rule_desc in ex_rules.items():
                        if "avoid" in rule_key or "max" in rule_key or "isometric" in rule_key:
                            forbidden_exercises = {
                                "isometric": ["plank", "wall sit", "static hold"],
                                "high_intensity": ["hiit", "sprint", "burpee", "jump"],
                                "high_impact": ["running", "jumping", "jump rope"],
                                "breath_holding": ["valsalva", "heavy lifting"],
                                "valsalva": ["valsalva", "breath holding", "breath-holding"],
                            }
                            keyword_key = rule_key.removeprefix("avoid_")
                            keywords = forbidden_exercises.get(keyword_key, [])
                            for keyword in keywords:
                                if keyword in plan_content:
                                    checks.append(SafetyCheck(
                                        check_name=f"{matched_condition}_{rule_key}",
                                        passed=False,
                                        message=f"Exercise plan contains {keyword} which violates {condition} restriction",
                                        severity=RiskLevel.HIGH
                                    ))
                                    risk_factors.append(RiskFactor(
                                        factor=f"{matched_condition}_{rule_key}",
                                        category="medical",
                                        severity=RiskLevel.HIGH,
                                        description=f"Exercise plan contains {keyword} which violates {condition} restriction",
                                        recommendation=f"Modify plan to comply with {rule_desc}"
                                    ))

                    # Execute the existing explicit threshold only with mg/dL input.
                    if "avoid_vigorous_if_below_100" in ex_rules:
                        glucose = user_metadata.get("current_glucose_mg_dl")
                        if glucose is None and str(user_metadata.get("blood_glucose_unit", "")).lower() in {"mg/dl", "mgdl"}:
                            glucose = user_metadata.get("blood_glucose")
                        try:
                            glucose = float(glucose)
                        except (ValueError, TypeError):
                            glucose = None
                        vigorous = any(term in plan_content for term in ("vigorous", "high intensity", "high_intensity", "hiit", "sprint", "intensity:high", "intensity:very_high"))
                        if glucose is not None and math.isfinite(glucose) and glucose < 100 and vigorous:
                            name = f"{matched_condition}_avoid_vigorous_if_below_100"
                            message = f"Vigorous exercise with glucose {glucose:g}mg/dL violates the configured threshold"
                            checks.append(SafetyCheck(check_name=name, passed=False, message=message, severity=RiskLevel.HIGH))
                            risk_factors.append(RiskFactor(factor=name, category="medical", severity=RiskLevel.HIGH,
                                                           description=message, recommendation="Avoid vigorous exercise under the configured glucose threshold"))

                    # ``max_heart_rate`` is configured as an age-dependent
                    # expression.  Compare it when the plan carries a measured
                    # or planned heart-rate value.
                    if "max_heart_rate" in ex_rules:
                        heart_rate = next(
                            (
                                plan.get(key)
                                for key in (
                                    "max_heart_rate",
                                    "heart_rate",
                                    "peak_heart_rate",
                                )
                                if plan.get(key) is not None
                            ),
                            None,
                        )
                        age = user_metadata.get("age")
                        try:
                            heart_rate = float(heart_rate)
                            age = float(age)
                            max_heart_rate = 220.0 - age * 0.7
                        except (TypeError, ValueError):
                            heart_rate = age = None
                            max_heart_rate = None
                        if (
                            heart_rate is not None
                            and max_heart_rate is not None
                            and heart_rate > max_heart_rate
                        ):
                            checks.append(SafetyCheck(
                                check_name=f"{matched_condition}_max_heart_rate",
                                passed=False,
                                message=(
                                    f"Heart rate {heart_rate:g} exceeds the calculated limit "
                                    f"{max_heart_rate:g}"
                                ),
                                severity=RiskLevel.HIGH
                            ))
                            risk_factors.append(RiskFactor(
                                factor=f"{matched_condition}_max_heart_rate",
                                category="medical",
                                severity=RiskLevel.HIGH,
                                description=(
                                    f"Heart rate {heart_rate:g} exceeds the calculated limit "
                                    f"{max_heart_rate:g}"
                                ),
                                recommendation="Reduce intensity and seek medical clearance"
                            ))

        return checks, risk_factors

    def _check_environment_safety(
        self,
        plan: Dict[str, Any],
        plan_type: str,
        environment: Dict[str, Any]
    ) -> tuple:
        """Check plan against environmental factors"""
        checks = []
        risk_factors = []

        weather = environment.get("weather", {})
        condition = weather.get("condition", "clear")
        temperature = weather.get("temperature_c", 20)

        # Temperature-based checks for outdoor exercise
        if plan_type == "exercise":
            if temperature > 35:
                risk_factors.append(RiskFactor(
                    factor="high_temperature_exercise",
                    category="environmental",
                    severity=RiskLevel.HIGH,
                    description=f"High temperature ({temperature} degrees Celsius) increases heat stress risk",
                    recommendation="Exercise indoors or in early morning/late evening"
                ))
            elif temperature < 5:
                risk_factors.append(RiskFactor(
                    factor="cold_temperature_exercise",
                    category="environmental",
                    severity=RiskLevel.MODERATE,
                    description=f"Cold temperature ({temperature} degrees Celsius) increases cardiovascular strain",
                    recommendation="Warm up thoroughly, dress in layers"
                ))

            if condition == "rainy" or condition == "icy":
                risk_factors.append(RiskFactor(
                    factor="inclement_weather",
                    category="environmental",
                    severity=RiskLevel.MODERATE,
                    description=f"{condition} weather increases slip/fall risk",
                    recommendation="Move exercise indoors or choose safe surfaces"
                ))

        # Diet considerations for environment
        if plan_type == "diet":
            if temperature > 30:
                # Suggest lighter meals
                checks.append(SafetyCheck(
                    check_name="hot_weather_hydration",
                    passed=True,
                    message="Consider increased fluid intake for hot weather"
                ))

        return checks, risk_factors


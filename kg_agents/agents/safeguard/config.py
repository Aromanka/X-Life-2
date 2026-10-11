def get_DIET_SAFETY_RULES(RiskLevel):
    return {
        "min_calories": {
            "value": 1200,
            "message": "Daily calories too low",
            "severity": RiskLevel.HIGH
        },
        "max_calories": {
            "value": 4000,
            "message": "Daily calories too high",
            "severity": RiskLevel.MODERATE
        },
        "min_protein_ratio": {
            "value": 0.10,
            "message": "Protein ratio too low (need adequate protein)",
            "severity": RiskLevel.MODERATE
        },
        "max_fat_ratio": {
            "value": 0.40,
            "message": "Fat ratio too high",
            "severity": RiskLevel.MODERATE
        },
        "single_meal_calories": {
            "value": 1500,
            "message": "Single meal calorie too high",
            "severity": RiskLevel.LOW
        }
    }


def get_EXERCISE_SAFETY_RULES(RiskLevel):
    return {
    "max_daily_duration_beginner": {
        "value": 30,
        "message": "Exercise duration too long for beginner",
        "severity": RiskLevel.HIGH
    },
    "max_daily_duration_intermediate": {
        "value": 60,
        "message": "Exercise duration too long",
        "severity": RiskLevel.MODERATE
    },
    "max_daily_duration_advanced": {
        "value": 120,
        "message": "Exercise duration excessive",
        "severity": RiskLevel.LOW
    },
    "min_rest_between_hiit": {
        "value": 48,
        "message": "HIIT sessions too frequent (need rest days)",
        "severity": RiskLevel.HIGH
    },
    "max_weekly_sessions": {
        "value": 7,
        "message": "Daily exercise without rest (need rest days)",
        "severity": RiskLevel.MODERATE
    }
}


def get_CONDITION_RESTRICTIONS():
    return {
    "diabetes": {
        "diet": {
            "avoid_high_sugar": "High sugar foods",
            "avoid_irregular_meals": "Irregular meal timing"
        },
        "exercise": {
            "avoid_vigorous_if_below_100": "Vigorous exercise with blood sugar < 100mg/dL",
            "avoid_late_exercise": "Late night exercise (hypoglycemia risk)"
        }
    },
    "hypertension": {
        "diet": {
            "max_sodium": 2300,
            "avoid_high_sodium": "High sodium foods"
        },
        "exercise": {
            "avoid_isometric": "Isometric exercises (heavy static holds)",
            "avoid_valsalva": "Breath holding during exercise"
        }
    },
    "heart_disease": {
        "exercise": {
            "max_heart_rate": "220 - age * 0.7",
            "avoid_high_intensity": "High intensity exercise",
            "require_clearance": "Medical clearance required"
        }
    },
    "obesity": {
        "exercise": {
            "avoid_high_impact": "High impact exercises",
            "start_low": "Low impact, gradual progression"
        }
    },
    "arthritis": {
        "exercise": {
            "avoid_high_impact": "Running, jumping",
            "prefer_low_impact": "Swimming, cycling"
        }
    }
}

# Deterministic physiological constraints are part of the safety gate.  Keep
# them enabled by default so a semantic-model outage cannot silently bypass
# hard thresholds.
ENABLE_RULE_BASED_CHECKS = True

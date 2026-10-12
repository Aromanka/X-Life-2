import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Import your pipelines
from pipeline.diet_pipeline import DietPipeline
from pipeline.exer_pipeline import ExercisePipeline
from agents.safeguard.assessor import SafeguardAgent
from core.world_model_scorer import WorldModelScorer

# --- Configuration ---
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Service Initialization (Load Models Once) ---
logger.info("Initializing Global AI Agents...")

# These instantiations will trigger any necessary model loading via the singletons
# in your core/llm/client.py and core/neo4j logic.
world_model_scorer = WorldModelScorer()
diet_service = DietPipeline(world_model_scorer=world_model_scorer.bind('diet'))
exercise_service = ExercisePipeline(world_model_scorer=world_model_scorer.bind('exercise'))
safeguard_service = SafeguardAgent()

logger.info("Agents initialized and ready.")


# --- Helper: Error Handling ---
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Server Error: {str(e)}", exc_info=True)
    return jsonify({"error": str(e), "type": type(e).__name__}), 500


# --- Endpoint 1: Generate Diet Plans ---
@app.route('/api/v1/diet/generate', methods=['POST'])
def generate_assess_diet():
    data = request.json
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    try:
        # Extract parameters with defaults matching your CLI args
        output = diet_service.generate(
            user_metadata=data.get("user_metadata", {}),
            environment=data.get("environment", {}),
            user_requirement=data.get("user_requirement", {}),
            user_query=data.get("user_query", None),
            num_base_plans=data.get("num_base_plans", 3),
            num_variants=data.get("num_variants", 3),
            min_scale=data.get("min_scale", 0.5),
            max_scale=data.get("max_scale", 1.5),
            meal_type=data.get("meal_type", "lunch"),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.92),
            # ``top_k`` controls generation sampling; ``top_k_selection``
            # controls the final number of safe recommendations.
            top_k=data.get("top_k", 50),
            top_k_selection=data.get("top_k_selection", 3),
            use_vector=data.get("use_vector", False),
            rag_topk=data.get("rag_topk", 3),
            output_path=data.get("output_path", "plan.json"),
            world_model_scorer=world_model_scorer.bind('diet', data.get('world_model_context'))
        )
        return jsonify(output.to_dict())

    except Exception as e:
        logger.error(f"Diet Generation Failed: {e}")
        return jsonify({"error": str(e)}), 500


# --- Endpoint 2: Generate Exercise Plans ---
@app.route('/api/v1/exercise/generate', methods=['POST'])
def generate_assess_exercise():
    data = request.json
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    try:
        output = exercise_service.generate(
            user_metadata=data.get("user_metadata", {}),
            environment=data.get("environment", {}),
            user_requirement=data.get("user_requirement", {}),
            user_query=data.get("user_query", None),
            num_base_plans=data.get("num_base_plans", 3),
            num_variants=data.get("num_variants", 3),
            min_scale=data.get("min_scale", 0.7),
            max_scale=data.get("max_scale", 1.3),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.92),
            # ``top_k`` controls generation sampling; ``top_k_selection``
            # controls the final number of safe recommendations.
            top_k=data.get("top_k", 50),
            top_k_selection=data.get("top_k_selection", 3),
            meal_timing=data.get("meal_timing", ""),
            use_vector=data.get("use_vector", False),
            rag_topk=data.get("rag_topk", 3),
            output_path=data.get("output_path", "exer_plan.json"),
            world_model_scorer=world_model_scorer.bind('exercise', data.get('world_model_context'))
        )
        return jsonify(output.to_dict())

    except Exception as e:
        logger.error(f"Exercise Generation Failed: {e}")
        return jsonify({"error": str(e)}), 500


# --- Endpoint 1b: Generate Diet Plans Only (No Assessment) ---
@app.route('/api/v1/diet/generate-only', methods=['POST'])
def generate_diet_only():
    """
    Generate diet plans WITHOUT safety assessment.
    Returns plans that can be assessed separately via /api/v1/safety/evaluate
    """
    data = request.json
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    try:
        output = diet_service.generate_only(
            user_metadata=data.get("user_metadata", {}),
            environment=data.get("environment", {}),
            user_requirement=data.get("user_requirement", {}),
            user_query=data.get("user_query", None),
            num_base_plans=data.get("num_base_plans", 3),
            num_variants=data.get("num_variants", 3),
            min_scale=data.get("min_scale", 0.5),
            max_scale=data.get("max_scale", 1.5),
            meal_type=data.get("meal_type", "lunch"),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.92),
            top_k=data.get("top_k", 50),
            use_vector=data.get("use_vector", False),
            rag_topk=data.get("rag_topk", 3)
        )
        return jsonify(output.to_dict())

    except Exception as e:
        logger.error(f"Diet Generate-Only Failed: {e}")
        return jsonify({"error": str(e)}), 500


# --- Endpoint 2b: Generate Exercise Plans Only (No Assessment) ---
@app.route('/api/v1/exercise/generate-only', methods=['POST'])
def generate_exercise_only():
    """
    Generate exercise plans WITHOUT safety assessment.
    Returns plans that can be assessed separately via /api/v1/safety/evaluate
    """
    data = request.json
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    try:
        output = exercise_service.generate_only(
            user_metadata=data.get("user_metadata", {}),
            environment=data.get("environment", {}),
            user_requirement=data.get("user_requirement", {}),
            user_query=data.get("user_query", None),
            num_base_plans=data.get("num_base_plans", 3),
            num_variants=data.get("num_variants", 3),
            min_scale=data.get("min_scale", 0.7),
            max_scale=data.get("max_scale", 1.3),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.92),
            top_k=data.get("top_k", 50),
            meal_timing=data.get("meal_timing", ""),
            use_vector=data.get("use_vector", False),
            rag_topk=data.get("rag_topk", 3)
        )
        return jsonify(output.to_dict())

    except Exception as e:
        logger.error(f"Exercise Generate-Only Failed: {e}")
        return jsonify({"error": str(e)}), 500


# --- Endpoint 3: Standalone Safety Evaluation ---
@app.route('/api/v1/safety/evaluate', methods=['POST'])
def evaluate_safety():
    """
    Evaluates a specific plan object (diet or exercise) without generating new ones.
    """
    data = request.json
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    required_fields = ["plan", "plan_type", "user_metadata"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    try:
        assessment = safeguard_service.assess(
            plan=data["plan"],
            plan_type=data["plan_type"],  # 'diet' or 'exercise'
            user_metadata=data["user_metadata"],
            environment=data.get("environment", {})
        )

        # assessment is a Pydantic model, use model_dump()
        return jsonify(assessment.model_dump())

    except Exception as e:
        logger.error(f"Safety Assessment Failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Host on 0.0.0.0 to make it accessible to network ports
    port = int(os.environ.get("PORT", 5000))
    print(f"Server starting on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)

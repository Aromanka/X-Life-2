from typing import List, Dict, Any, Optional
from .driver import Neo4jClient, get_neo4j
from config_loader import get_config
import os


# Shared embedding model cache
_embedding_model = None
_embedding_dim = None


def get_embedding_model():
    """Lazy load and cache the embedding model"""
    global _embedding_model, _embedding_dim
    if _embedding_model is not None:
        return _embedding_model

    from sentence_transformers import SentenceTransformer
    config = get_config()
    local_model_path = config.get("local_emb_path", None)

    if local_model_path and os.path.exists(local_model_path):
        _embedding_model = SentenceTransformer(local_model_path, local_files_only=True)
    else:
        raise ValueError('Set local_emb_path to an existing local embedding model directory.')

    _embedding_dim = _embedding_model.get_sentence_embedding_dimension()
    return _embedding_model


def get_embedding(text: str) -> List[float]:
    """Get embedding for a text string"""
    model = get_embedding_model()
    return model.encode(text).tolist()


class KnowledgeGraphQuery:

    def __init__(self, client: Optional[Neo4jClient] = None):
        self.client = client or get_neo4j()

    def search_exact_entities(self, name: str) -> List[Dict[str, Any]]:
        """Case-insensitive exact entity name; parameters are never interpolated."""
        return self._safe_query("""
            MATCH (head)-[r]-(tail)
            WHERE toLower(head.name) = toLower($name)
            RETURN head.name AS head, type(r) AS rel_type, tail.name AS tail,
                   coalesce(r.condition, r.context, '') AS condition
            LIMIT 40
        """, {"name": name})

    # dietary querires

    def _safe_query(self, pattern: str, params: dict):
        """Execute a query, return empty list if fails"""
        try:
            return self.client.query(pattern, params)
        except Exception as e:
            print(f"[WARN] Query failed: {e}")
            return []

    def query_foods_by_disease(self, disease: str) -> List[Dict[str, Any]]:
        pattern = """
        MATCH (d)-[r]->(f)
        WHERE toLower(d.name) = toLower($disease)
        RETURN f.name as food, type(r) as relation
        LIMIT 20
        """
        return self._safe_query(pattern, {"disease": disease})

    def query_foods_for_condition(self, condition: str) -> List[Dict[str, Any]]:
        pattern = """
        MATCH (n)-[r]->(m)
        WHERE toLower(n.name) CONTAINS toLower($condition)
           OR toLower($condition) CONTAINS toLower(n.name)
        RETURN n.name as entity, type(r) as relation, m.name as target
        """
        return self._safe_query(pattern, {"condition": condition})

    def query_dietary_restrictions(self, disease: str) -> List[Dict[str, Any]]:
        pattern = """
        MATCH (d)-[r]->(rstr)
        WHERE toLower(d.name) = toLower($disease)
        RETURN rstr.name as entity, type(r) as relation
        LIMIT 20
        """
        return self._safe_query(pattern, {"disease": disease})

    def query_nutrient_advice(self, disease: str) -> List[Dict[str, Any]]:
        pattern = """
        MATCH (n)-[r]->(d)
        WHERE toLower(d.name) = toLower($disease)
        RETURN n.name as entity, type(r) as relation
        LIMIT 20
        """
        return self._safe_query(pattern, {"disease": disease})

    def query_food_benefits(self, food: str) -> List[Dict[str, Any]]:
        pattern = """
        MATCH (f)-[r]->(b)
        WHERE toLower(f.name) CONTAINS toLower($food)
        RETURN b.name as entity, type(r) as relation
        LIMIT 10
        """
        return self._safe_query(pattern, {"food": food})

    def query_food_risks(self, food: str) -> List[Dict[str, Any]]:
        pattern = """
        MATCH (f)-[r]->(rsk)
        WHERE toLower(f.name) CONTAINS toLower($food)
        RETURN rsk.name as entity, type(r) as relation
        LIMIT 10
        """
        return self._safe_query(pattern, {"food": food})

    def query_food_conflicts(self, food: str) -> List[Dict[str, Any]]:
        pattern = """
        MATCH (f)-[r]->(d)
        WHERE toLower(f.name) CONTAINS toLower($food)
        RETURN d.name as entity, type(r) as relation
        LIMIT 10
        """
        return self._safe_query(pattern, {"food": food})

    # exercise query

    def query_exercise_for_condition(self, condition: str) -> List[Dict[str, Any]]:
        pattern = """
        MATCH (e)-[r]->(c)
        WHERE toLower(c.name) CONTAINS toLower($condition)
           OR toLower($condition) CONTAINS toLower(c.name)
        RETURN e.name as entity, type(r) as relation, labels(e) as labels
        LIMIT 20
        """
        return self._safe_query(pattern, {"condition": condition})

    def query_exercise_avoid_for_condition(self, condition: str) -> List[Dict[str, Any]]:
        pattern = """
        MATCH (e)-[r]->(c)
        WHERE toLower(c.name) CONTAINS toLower($condition)
        RETURN e.name as entity, type(r) as relation, labels(e) as labels
        LIMIT 20
        """
        return self._safe_query(pattern, {"condition": condition})

    def query_exercise_risks(
        self,
        exercise: str = None,
        condition: str = None
    ) -> List[Dict[str, Any]]:
        if condition:
            pattern = """
            MATCH (e)-[r]->(c)
            WHERE toLower(c.name) CONTAINS toLower($condition)
            RETURN e.name as entity, type(r) as relation, labels(e) as labels
            LIMIT 20
            """
            return self._safe_query(pattern, {"condition": condition})
        elif exercise:
            pattern = """
            MATCH (e)-[r]->(target)
            WHERE toLower(e.name) CONTAINS toLower($exercise)
            RETURN target.name as entity, type(r) as relation, labels(target) as labels
            LIMIT 10
            """
            return self._safe_query(pattern, {"exercise": exercise})
        return []

    def query_exercise_benefits(self, exercise: str) -> List[Dict[str, Any]]:
        pattern = """
        MATCH (e)-[r]->(b)
        WHERE toLower(e.name) CONTAINS toLower($exercise)
        RETURN b.name as entity, type(r) as relation, labels(b) as labels
        LIMIT 10
        """
        return self._safe_query(pattern, {"exercise": exercise})

    def query_exercise_targets_muscle(self, exercise: str) -> List[Dict[str, Any]]:
        pattern = """
        MATCH (e)-[r]->(m)
        WHERE toLower(e.name) CONTAINS toLower($exercise)
        RETURN m.name as entity, type(r) as relation, labels(m) as labels
        LIMIT 10
        """
        return self._safe_query(pattern, {"exercise": exercise})

    def query_exercise_duration(self, exercise: str) -> List[Dict[str, Any]]:
        pattern = """
        MATCH (e)-[r]->(d)
        WHERE toLower(e.name) CONTAINS toLower($exercise)
        RETURN d.name as entity, type(r) as relation, labels(d) as labels
        LIMIT 10
        """
        return self._safe_query(pattern, {"exercise": exercise})

    def query_exercise_frequency(self, exercise: str) -> List[Dict[str, Any]]:
        pattern = """
        MATCH (e)-[r]->(f)
        WHERE toLower(e.name) CONTAINS toLower($exercise)
        RETURN f.name as entity, type(r) as relation, labels(f) as labels
        LIMIT 10
        """
        return self._safe_query(pattern, {"exercise": exercise})

    def query_exercise_substitutes(self, exercise: str) -> List[Dict[str, Any]]:
        pattern = """
        MATCH (e)-[r]->(s)
        WHERE toLower(e.name) CONTAINS toLower($exercise)
        RETURN s.name as entity, type(r) as relation, labels(s) as labels
        LIMIT 10
        """
        return self._safe_query(pattern, {"exercise": exercise})

    def query_exercise_by_type(
        self,
        exercise_type: str,
        intensity: str = None
    ) -> List[Dict[str, Any]]:
        pattern = """
        MATCH (e)
        WHERE toLower(e.type) = toLower($type)
        """
        params = {"type": exercise_type}

        if intensity:
            pattern += " AND toLower(e.intensity) = toLower($intensity)"
            params["intensity"] = intensity

        pattern += """
        RETURN e.name as name, labels(e) as labels
        LIMIT 30
        """
        return self._safe_query(pattern, params)

    def query_all_exercises(self, limit: int = 50) -> List[Dict[str, Any]]:
        pattern = """
        MATCH (e)
        RETURN e.name as name, labels(e) as labels
        LIMIT $limit
        """
        return self._safe_query(pattern, {"limit": limit})

    # general queries

    def search_entities(self, keyword: str, limit: int = 20) -> List[Dict[str, Any]]:
        return self.client.search_by_keyword(keyword, score_threshold=0.2)

    def search_similar_entities(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Vector-based semantic search using Neo4j Vector Index (GraphRAG)

        Args:
            query_text: Natural language query text
            top_k: Number of similar entities to return

        Returns:
            List of similar entities with similarity scores
        """
        try:
            query_vector = get_embedding(query_text)
            cypher = """
            CALL db.index.vector.queryNodes('node_embedding_index', $top_k, $query_vector)
            YIELD node, score
            RETURN node.name AS name, score, elementId(node) AS id
            ORDER BY score DESC
            """
            results = self.client.query(cypher, {"query_vector": query_vector, "top_k": top_k})
            return [dict(record) for record in results if results]
        except Exception as e:
            print(f"[WARN] Vector search failed: {e}")
            return []

    def get_entity_info(self, name: str) -> Optional[Dict[str, Any]]:
        node = self.client.get_node_by_name(name)
        if node:
            neighbors = self.client.get_neighbors(name)
            return {"node": node, "neighbors": neighbors}
        return None


_kg_query: Optional[KnowledgeGraphQuery] = None


def get_kg_query() -> KnowledgeGraphQuery:
    global _kg_query
    if _kg_query is None:
        _kg_query = KnowledgeGraphQuery()
    return _kg_query

"""Optional database adapter; no connection on import or construction."""
class Neo4jGraph:
    def __init__(self, uri, username, password):
        self.settings = uri, username, password

    def snapshot(self):
        from neo4j import GraphDatabase
        from knowledge import KnowledgeGraph
        uri, user, password = self.settings
        with GraphDatabase.driver(uri, auth=(user,password)) as driver:
            with driver.session() as session:
                rows = session.run('MATCH (h)-[r]->(t) RETURN h.name AS head, type(r) AS relation, t.name AS tail, r.source AS source')
                return KnowledgeGraph([dict(r) for r in rows], 'neo4j')

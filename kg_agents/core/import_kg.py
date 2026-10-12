"""
Import knowledge graph data from JSON/CSV files into Neo4j database.
Run after build_kg_deepseek.py has generated the output files.
"""
import os
import json
import glob
from neo4j import GraphDatabase
from config_loader import NEO4J_URI, NEO4J_AUTH

# load config

OUTPUT_HISTORY_DIR = "output_history"
INPUT_DIR = "data"


RELATION_LABEL_MAP = {
    # Disease
    "Diet_Disease": ("Food", "Disease"),
    "Food_Disease": ("Food", "Disease"),
    "Nutrient_Disease": ("Nutrient", "Disease"),
    "Restriction_Disease": ("Restriction", "Disease"),
    "Contraindication_Food": ("Food", "Disease"),
    "Interaction_Food": ("Food", "Disease"),

    # Diet
    "Food_Diet": ("Food", "Diet"),

    # Food
    "Amount_Food": ("Amount", "Food"),
    "Frequency_Food": ("Frequency", "Food"),
    "Method_Food": ("Method", "Food"),

    # Food
    "Benefit_Food": ("Food", "Benefit"),
    "Risk_Food": ("Food", "Risk"),
}


def infer_entity_label(name: str, position: str, relation: str) -> str:
    name_lower = name.lower()

    if relation in RELATION_LABEL_MAP:
        labels = RELATION_LABEL_MAP[relation]
        return labels[0] if position == "head" else labels[1]

    diseases = ["diabetes", "hypertension", "heart disease", "obesity", "cancer",
                 "asthma", "arthritis", "anemia", "gout", "kidney disease"]
    if any(d in name_lower for d in diseases):
        return "Disease"

    nutrients = ["protein", "carbohydrate", "fat", "fiber", "vitamin",
                  "mineral", "calcium", "iron", "zinc", "sodium", "potassium"]
    if any(n in name_lower for n in nutrients):
        return "Nutrient"

    return "Entity"


# core logic
driver = GraphDatabase.driver(NEO4J_URI(), auth=NEO4J_AUTH())


def create_indexes(session):
    indexes = [
        "CREATE FULLTEXT INDEX search_index IF NOT EXISTS FOR (n:Entity) ON EACH [n.name]",
        "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (n:Entity) ON (n.name)",
    ]
    for idx in indexes:
        try:
            session.run(idx)
        except Exception as e:
            print(f"Failed/Skip creating index: {e}")


def clear_database(session):
    confirm = input("Warn: Clear current data?(y/n): ")
    if confirm.lower() != 'y':
        print("Denied clear. ")
        return False
    session.run("MATCH (n) DETACH DELETE n")
    print("Database cleared.")
    return True


def import_json_triplets(session, json_path):
    print(f"Importing: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        triplets = json.load(f)

    if not triplets:
        print(f"  Skip empty file...")
        return 0

    count = 0
    for t in triplets:
        head = t.get('head', '').strip()
        relation = t.get('relation', '').strip()
        tail = t.get('tail', '').strip()
        source = t.get('source', '')
        context = t.get('context') or ''

        if not head or not relation or not tail:
            continue

        try:
            rel_type = relation.replace(' ', '_').replace('-', '_')

            # infer entity label
            head_label = infer_entity_label(head, "head", relation)
            tail_label = infer_entity_label(tail, "tail", relation)

            # create entity & relation
            session.run(f"""
                MERGE (h:Entity {{name: $head}})
                SET h:{head_label}
                MERGE (t:Entity {{name: $tail}})
                SET t:{tail_label}
                MERGE (h)-[r:`{rel_type}` {{source: $source, context: $context}}]->(t)
            """, head=head, tail=tail, relation=relation, source=source, context=context)
            count += 1
        except Exception as e:
            print(f"  Failed to import: {head} -[{relation}]-> {tail}: {e}")

    print(f"  Successfully import {count} relations")
    return count


def import_csv_triplets(session, csv_path):
    import pandas as pd

    print(f"Importing csv: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"  Empty file skip")
        return 0

    count = 0
    for _, row in df.iterrows():
        head = str(row.get('head', '')).strip()
        relation = str(row.get('relation', '')).strip()
        tail = str(row.get('tail', '')).strip()
        source = str(row.get('source', '')).strip()
        context = row.get('context', '')
        context = '' if pd.isna(context) else str(context)

        if not head or not relation or not tail or head == 'nan':
            continue

        try:
            rel_type = relation.replace(' ', '_').replace('-', '_')

            # infer entity label
            head_label = infer_entity_label(head, "head", relation)
            tail_label = infer_entity_label(tail, "tail", relation)

            session.run(f"""
                MERGE (h:Entity {{name: $head}})
                SET h:{head_label}
                MERGE (t:Entity {{name: $tail}})
                SET t:{tail_label}
                MERGE (h)-[r:`{rel_type}` {{source: $source, context: $context}}]->(t)
            """, head=head, tail=tail, relation=relation, source=source, context=context)
            count += 1
        except Exception as e:
            print(f"  Failed to import: {head} -[{relation}]-> {tail}: {e}")

    print(f"  Successfully imported {count} relations")
    return count


def import_from_output_history(session):
    if not os.path.exists(OUTPUT_HISTORY_DIR):
        print(f"  Directory not found: {OUTPUT_HISTORY_DIR}")
        return 0

    json_files = glob.glob(os.path.join(OUTPUT_HISTORY_DIR, "**/*.json"), recursive=True)
    csv_files = glob.glob(os.path.join(OUTPUT_HISTORY_DIR, "**/*.csv"), recursive=True)

    total = 0
    for json_file in json_files:
        total += import_json_triplets(session, json_file)

    for csv_file in csv_files:
        total += import_csv_triplets(session, csv_file)

    return total


def import_from_directory(session, directory):
    json_files = glob.glob(os.path.join(directory, "*.json"))
    csv_files = glob.glob(os.path.join(directory, "*.csv"))

    total = 0
    for json_file in json_files:
        total += import_json_triplets(session, json_file)

    for csv_file in csv_files:
        total += import_csv_triplets(session, csv_file)

    return total


def show_stats(session):
    print("\nDatabase statistics:")

    result = session.run("MATCH (n) RETURN count(n) as count")
    entity_count = result.single()["count"]
    print(f"  Entity count: {entity_count}")

    print("  Entity label distribution:")
    result = session.run("""
        MATCH (n)
        RETURN labels(n) as labels, count(n) as count
        ORDER BY count DESC
    """)
    for record in result:
        print(f"    {record['labels']}: {record['count']}")

    result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
    rel_count = result.single()["count"]
    print(f"  Relation count: {rel_count}")

    print("  Relation type distribution:")
    result = session.run("""
        MATCH ()-[r]->()
        RETURN type(r) as type, count(r) as count
        ORDER BY count DESC
    """)
    for record in result:
        print(f"    {record['type']}: {record['count']}")


def main():
    print("Neo4j KG import:")

    with driver.session() as session:
        print("Choose mode:")
        print("1. Import from output_history folder")
        print("2. Import from specific folder")
        print("3. Show database statistics")
        print("4. Clear database and exit")

        choice = input("\nChoose (1-4): ").strip()

        if choice == "1":
            create_indexes(session)
            total = import_from_output_history(session)
            print(f"\n>>> Total imported {total} relations")
            show_stats(session)

        elif choice == "2":
            directory = input("Enter directory path: ").strip()
            if os.path.exists(directory):
                create_indexes(session)
                total = import_from_directory(session, directory)
                print(f"\n>>> Total imported {total} relations")
                show_stats(session)
            else:
                print(f"  Directory not found: {directory}")

        elif choice == "3":
            show_stats(session)

        elif choice == "4":
            if clear_database(session):
                print("Database cleared")

        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()

import os
import openai
import pywikibot
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv
import re
import requests
from queue import Queue
openai_api_key = os.getenv("OPENAI_API_KEY")
pywikibot.config.socket_timeout = 30
site = pywikibot.Site("wikidata", "wikidata")
repo = site.data_repository()

def sanitize_input(text):
    """Remove unwanted prefixes and trim text."""
    return re.sub(r'^[\d\.\-]+\s*', '', text).strip()

def robust_request(item_id):
    """Fetch a single item from Wikidata by item ID."""
    try:
        item = pywikibot.ItemPage(repo, item_id)
        item.get()
        print(f"Successfully fetched Wikidata item: {item_id}")
        return item if item.exists() else None
    except Exception as e:
        print(f"Failed to fetch item '{item_id}' due to error: {e}")
        return None

def fetch_label_by_id(entity_id):
    try:
        page = pywikibot.PropertyPage(repo, entity_id) if entity_id.startswith('P') else pywikibot.ItemPage(repo, entity_id)
        page.get(force=True)
        label = page.labels.get('en', 'No label found')
        print(f"Label for {entity_id}: {label}")
        return label
    except Exception as e:
        print(f"Error fetching label for ID {entity_id}: {e}")
        return "Invalid ID"

def paraphrase_subject(subject_label):
    prompt = (
        "Alan Turing is also known as:\n"
        "- The father of computing\n"
        "- A pioneer in computer science\n"
        "- The codebreaker of World War II\n"
        "\n"
        f"{subject_label} is also known as:"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Generate paraphrases for the subject."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    paraphrases_text = response.choices[0].message['content'].strip()
    paraphrases = re.split(r'\s*\n+', paraphrases_text)
    paraphrases = [sanitize_input(p) for p in paraphrases if is_valid_paraphrase(p)]
    print(f"Subject paraphrases for '{subject_label}': {paraphrases}")
    return paraphrases

def paraphrase_relation(relation_label):
    prompt = (
        f"'notable work' may be described as:\n"
        "- A work of great value\n"
        "- A work of importance\n"
        "'notable work' refers to:\n"
        "- Significant achievements\n"
        "- Important contributions\n"
        "please describe 'notable work' in a few words:\n"
        "- Key accomplishments\n"
        "- Major works\n"
        "\n"
        f"'{relation_label}' may be described as:"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Generate paraphrases for the relation."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    paraphrases_text = response.choices[0].message['content'].strip()
    paraphrases = re.split(r'\s*\n+', paraphrases_text)
    paraphrases = [sanitize_input(p) for p in paraphrases if is_valid_paraphrase(p)]
    print(f"Relation paraphrases for '{relation_label}': {paraphrases}")
    return paraphrases


def is_valid_paraphrase(paraphrase):
    valid = len(paraphrase.split()) > 1 or (len(paraphrase) > 1 and paraphrase.isalpha())
    if not valid:
        print(f"Invalid paraphrase discarded: {paraphrase}")
    return valid

def resolve_wikidata_id(paraphrases):
    wikipedia_site = pywikibot.Site('en', 'wikipedia')
    for paraphrase in paraphrases:
        print(f"Resolving paraphrase: {paraphrase}")
        search_page = wikipedia_site.search(paraphrase, total=1)
        for page in search_page:
            if page.exists():
                if page.isRedirectPage():
                    page = page.getRedirectTarget()
                if page.data_item():
                    wikidata_id = page.data_item().title()
                    print(f"Resolved to Wikidata ID: {wikidata_id} for paraphrase: {paraphrase}")
                    return wikidata_id
        print(f"No Wikidata ID found for paraphrase: {paraphrase}")
    return None

def query_model_with_few_shot(entity_label, relation_label):
    """Query the GPT-3.5 model for objects based on entity and relation label with DK object generation."""
    prompt = (
        "Provide information or state 'Don't know' for:\n"
        "Q: Monte Cremasco # country\n"
        "A: Italy\n"
        "Q: Johnny Depp # children\n"
        "A: Jack Depp, Lily-Rose Depp\n"
        "Q: Wolfgang Sauseng # employer\n"
        "A: University of Music and Performing Arts Vienna\n"
        "Q: Barack Obama # child\n"
        "A:\n"
        f"Q: {entity_label} # {relation_label}\n"
        "A:"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Answer the query or state 'Don't know'."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    answer = response.choices[0].message['content'].strip()
    answer = answer.replace('Category:', '').strip()
    print(f"Query result for '{entity_label} # {relation_label}': {answer}")
    return answer

def visualize_graph(graph):
    A = nx.nx_agraph.to_agraph(graph)
    A.graph_attr.update(
        splines='true', 
        rankdir='LR', 
        size='16,10',  
        fontsize=12,
        fontcolor='blue'
    )
    A.node_attr.update(
        shape='ellipse', 
        style='filled', 
        fillcolor='lightblue', 
        fontsize=14,
        height=0.6,  
        width=1.2   
    )
    
    A.edge_attr.update(
        fontsize=12, 
        fontcolor='black',
        color='red',  
        arrowsize=0.7  
    )
    A.layout(prog='dot')
    output_path = '/Users/maochuan/Desktop/graph.png'
    A.draw(output_path)
    print(f"Graph saved to {output_path}")

def fetch_initial_relations(wikidata_item):
    relations = []
    if not wikidata_item:
        return relations
    for claim in wikidata_item.claims:
        target_items = wikidata_item.claims[claim]
        for target_item in target_items:
            target = target_item.getTarget()
            if isinstance(target, pywikibot.ItemPage):
                relations.append((claim, target.title()))
    return relations

def generate_relations(entity_label):
    prompt = (
        "Q: Javier Culson\n"
        "A: participant of # place of birth # sex or gender # country of citizenship # occupation # family name # given name # educated at # sport # sports discipline competed in\n"
        "Q: René Magritte\n"
        "A: ethnic group # place of birth # place of death # sex or gender # spouse # country of citizenship # member of political party # native language # place of burial # cause of death # residence # family name # given name # manner of death # educated at # field of work # work location # represented by\n"
        "Q: Nadym\n"
        "A: country # capital of # coordinate location # population # area # elevation above sea level\n"
        "Q: Stryn\n"
        "A: significant event # head of government # country # capital # separated from\n"
        "Q: 1585\n"
        "A: said to be the same as # follows\n"
        "Q: Bornheim\n"
        "A: head of government # country # member of # coordinate location # population # area # elevation above sea level\n"
        "Q: Aló Presidente\n"
        "A: genre # country of origin # cast member # original network\n"
        f"Q: {entity_label}\n"
        "A:"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Generate relations for the entity."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    relations_text = response.choices[0].message['content'].strip()
    relations = [sanitize_input(r) for r in re.split(r'#\s*', relations_text)]
    print(f"Generated relations for '{entity_label}': {relations}")
    return relations


def construct_knowledge_graph(entity_id, max_depth=2, branch_limit=2):
    graph = nx.DiGraph()
    queue = Queue()
    queue.put((entity_id, 0))  # Enqueue the initial node and its depth

    while not queue.empty():
        current_id, current_depth = queue.get()
        if current_depth > max_depth:
            continue  # Skip processing if the current depth exceeds the maximum depth

        current_label = fetch_label_by_id(current_id)
        if not graph.has_node(current_label):
            graph.add_node(current_label)
            print(f"Added node: {current_label} at depth {current_depth}")

        if current_depth == max_depth:
            continue  # Do not expand nodes at the maximum depth

        paraphrases = paraphrase_subject(current_label)
        branches_created = 0

        for paraphrase in paraphrases:
            paraphrase_id = resolve_wikidata_id([paraphrase])
            if paraphrase_id:
                relations = fetch_initial_relations(robust_request(paraphrase_id))
                if not relations:
                    relations = [(None, relation) for relation in generate_relations(paraphrase)]
                for relation_id, target_id in relations:
                    if branches_created >= branch_limit:
                        break
                    relation_label = fetch_label_by_id(relation_id) if relation_id else relation_id
                    relation_paraphrases = paraphrase_relation(relation_label) if relation_id else [relation_label]
                    for relation_paraphrase in relation_paraphrases:
                        dk_object = query_model_with_few_shot(paraphrase, relation_paraphrase)
                        if dk_object.lower() != "don't know":
                            target_label = fetch_label_by_id(target_id)
                            if not graph.has_edge(current_label, target_label):
                                graph.add_edge(current_label, target_label, label=relation_paraphrase)
                                print(f"Added edge from '{current_label}' to '{target_label}' with relation '{relation_paraphrase}' at depth {current_depth}")
                                queue.put((target_id, current_depth + 1))  # Enqueue the new node with incremented depth
                                branches_created += 1
                            break  # Only add one relation paraphrase with the object

    return graph

def main():
    root_entity_id = "Q76"  # obama
    max_depth = 2
    graph = construct_knowledge_graph(root_entity_id, max_depth, branch_limit = 3)
    visualize_graph(graph)

if __name__ == "__main__":
    main()

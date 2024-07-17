import os
import openai
import pywikibot
import networkx as nx
import matplotlib.pyplot as plt
import re
from relations import our_relations
import requests
from queue import Queue
openai.api_key = os.getenv("OPENAI_API_KEY")
pywikibot.config.socket_timeout = 30
SITE = pywikibot.Site("wikidata", "wikidata")
REPO = SITE.data_repository()

def sanitize_input(text):
    """Remove unwanted prefixes and trim text."""
    return re.sub(r'^[\d\.\-]+\s*', '', text).strip()

def robust_request(item_id):
    """Fetch a single item from Wikidata by item ID."""
    try:
        item = pywikibot.ItemPage(REPO, item_id)
        item.get()
        print(f"Successfully fetched Wikidata item: {item_id}")
        return item if item.exists() else None
    except Exception as e:
        print(f"Failed to fetch item '{item_id}' due to error: {e}")
        return None

def fetch_label_by_id(entity_id):
    try:
        page = pywikibot.PropertyPage(REPO, entity_id) if entity_id.startswith('P') else pywikibot.ItemPage(REPO, entity_id)
        page.get(force=True)
        label = page.labels.get('en', 'No label found')
        print(f"Label for {entity_id}: {label}")
        return label
    except Exception as e:
        print(f"Error fetching label for ID {entity_id}: {e}")
        return "Invalid ID"

def paraphrase_subject(subject_label):
    prompt = (
        "Bill Clinton is also known as:\n"
        "- William Clinton\n"
        "- William Jefferson Clinton\n"
        "- The 42nd president of the United States\n"
        "\n"
        "United States of America is also known as:\n"
        "- United States\n"
        "- US\n"
        "- America\n"
        "\n"

        f"{subject_label} is also known as:"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Generate paraphrases for the subject in specific form."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    paraphrases_text = response.choices[0].message['content'].strip()
    paraphrases = re.split(r'\s*\n+', paraphrases_text)
    sanitized_paraphrases = []
    for p in paraphrases:
        if ":" in p:
            p = p.split(":")[1].strip()
        sanitized_paraphrase = sanitize_input(p)
        if is_valid_paraphrase_subject(sanitized_paraphrase):
            sanitized_paraphrases.append(sanitized_paraphrase)
    print(f"Subject paraphrases for '{subject_label}': {sanitized_paraphrases}")
    return sanitized_paraphrases

def paraphrase_relation(relation_label):
    instructions = [
        f"'{relation_label}' may be described as:",
        f"'{relation_label}' refers to:",
        "please describe '{}' in a few words:".format(relation_label)
    ]

    all_paraphrases = set()  # Use a set to avoid duplicates

    for instruction in instructions:
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
            f"{instruction}"
        )
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate paraphrases for the relation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        
        paraphrases_text = response['choices'][0]['message']['content'].strip()
        paraphrases = re.split(r'\s*\n+', paraphrases_text)
        valid_paraphrases = [sanitize_input(p) for p in paraphrases if is_valid_paraphrase_relation(p, instructions)]
        all_paraphrases.update(valid_paraphrases)  # Add to set to avoid duplicates

    print(f"Relation paraphrases for '{relation_label}': {list(all_paraphrases)}")
    return list(all_paraphrases)

def is_valid_paraphrase_relation(paraphrase, instructions):
    """ Check if the generated paraphrase is valid based on some criteria. """
    paraphrase_lower = paraphrase.lower()
    for instr in instructions:
        if instr.lower() in paraphrase_lower or paraphrase_lower in instr.lower() or "please paraphrase" in paraphrase_lower:
            return False
    return len(paraphrase) > 0 and not paraphrase_lower.startswith("error")


def is_valid_paraphrase_subject(paraphrase):
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

def generate_object(entity_label, relation_label):
    """Generate object based on entity and relation labels using a structured prompt for the GPT-3.5 model."""
    prompt = (
        "Q: Monte Cremasco # country\n"
        "A: Italy\n"
        "Q: Johnny Depp # children\n"
        "A: Jack Depp, Lily-Rose Depp\n"
        "Q: Wolfgang Sauseng # employer\n"
        "A: University of Music and Performing Arts Vienna\n"
        f"Q: {entity_label} # {relation_label}\n"
        "A: ?"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Answer the query in strict form: A: '{object}'"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        stop=["\nQ:", "\nA:"] 
    )
    answer = response.choices[0].message['content'].strip()
    
    # Ensure to extract content after "A:" and handle if "A:" is not found
    if "A:" in answer:
        real_object = answer.split("A:", 1)[1].strip()
    else:
        real_object = answer

    real_object = sanitize_input(real_object)
    print(f"Query result for '{entity_label} # {relation_label}': {real_object}")
    return real_object
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
    output_path = '/Users/maochuan/Desktop/graph5.png'
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


def construct_knowledge_graph(entity_label, max_depth=2, branch_limit=2):
    import networkx as nx
    from queue import Queue

    graph = nx.DiGraph()
    queue = Queue()
    queue.put((entity_label, 0))  # Enqueue the initial node and its depth
    print(f"Starting graph construction with root entity ID: {entity_label}")

    while not queue.empty():
        current_label, current_depth = queue.get()
        print(f"Processing entity: {current_label} at depth: {current_depth}")

        if current_depth > max_depth:
            print("Current depth exceeds max depth, skipping...")
            continue  # Skip processing if the current depth exceeds the maximum depth

        if not graph.has_node(current_label):
            graph.add_node(current_label)
            print(f"Added node: {current_label} at depth {current_depth}")

        if current_depth == max_depth:
            print(f"Reached maximum depth at node: {current_label}, not expanding further.")
            continue  # Do not expand nodes at the maximum depth

        paraphrases = paraphrase_subject(current_label)
        print(f"Paraphrases found for '{current_label}': {paraphrases}")
        branches_created = 0

        # Initialize dictionary to store relations for each paraphrase
        paraphrase_relations = {paraphrase: set() for paraphrase in paraphrases}

        # Collect all possible relations for each paraphrase
        for paraphrase in paraphrases:
            paraphrase_id = resolve_wikidata_id([paraphrase])
            print(f"Resolved Wikidata ID for paraphrase '{paraphrase}': {paraphrase_id}")

            if paraphrase_id:
                item = robust_request(paraphrase_id)
                relations = fetch_initial_relations(item)
                print(f"Initial relations fetched for paraphrase '{paraphrase}': {relations}")

                for rel_id, _ in relations:
                    paraphrase_relations[paraphrase].add(rel_id)

        # Calculate intersection of all relation sets and filter with our_relations
        valid_our_relations = {v for v in our_relations.values() if v}
        print(f"valid relations: {valid_our_relations}")
        val_paraphrased_relations = paraphrase_relations.values()
        print(f"paraphrase_relations.values(): {val_paraphrased_relations}")
        common_relation_ids = set.intersection(*paraphrase_relations.values(), valid_our_relations)
        print(f"Common relation IDs across all paraphrases and our_relations: {common_relation_ids}")
        if not common_relation_ids:
            print("No common relations found, generating new relations...")
            common_relation_labels = generate_relations(current_label)  
            print(f"Generated relations: {common_relation_ids}")
        else: 
            common_relation_labels = {fetch_label_by_id(rel_id) for rel_id in common_relation_ids}
        for relation_label in common_relation_labels:
            if branches_created >= branch_limit:
                print("Branch limit reached, not creating more branches.")
                break
            relation_paraphrases = paraphrase_relation(relation_label)
            print(f"Paraphrases for relation '{relation_label}': {relation_paraphrases}")
            for relation_paraphrase in relation_paraphrases:
                object = generate_object(current_label, relation_paraphrase)
                print(f"object: {object}")
                if not graph.has_edge(current_label, object):
                    graph.add_edge(current_label, object, label=relation_paraphrase)
                    print(f"Added edge from '{current_label}' to '{object}' with relation '{relation_paraphrase}' at depth {current_depth}")
                    queue.put((object, current_depth + 1))
                    branches_created += 1
                break

    print("Graph construction completed.")
    return graph



def main():
    # root_entity_id = "Q76" #obama
    # root_entity_label = fetch_label_by_id(root_entity_id)
    root_entity_label = "Maochuan Lu"
    max_depth = 2
    graph = construct_knowledge_graph(root_entity_label, max_depth, branch_limit = 2)
    visualize_graph(graph)

if __name__ == "__main__":
    main()

from flask import Flask, request, jsonify, render_template
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
from itertools import chain

app = Flask(__name__)

def parse_nfa_transitions(transitions):
    transition_dict = {}
    try:
        for trans in transitions.split(";"):
            trans = trans.strip()
            if not trans:
                continue
            parts = trans.split("=")
            if len(parts) != 2:
                print(f"Invalid transition format: {trans}")
                continue
            state_input, next_states = parts[0].strip(), parts[1].strip()
            if not next_states:
                print(f"Invalid transition with no next states: {trans}")
                continue

            state_parts = state_input.split(",")
            if len(state_parts) != 2:
                print(f"Invalid state,input format: {state_input}")
                continue

            state, input_char = state_parts[0].strip(), state_parts[1].strip()
            transition_dict.setdefault(state, {}).setdefault(input_char, set()).update(
                s.strip() for s in next_states.split(",") if s.strip()
            )

        return transition_dict
    except Exception as e:
        print(f"Transition parsing error: {e}")
        return None

def nfa_to_dfa(nfa_transitions, start_state, final_states):
    dfa_transitions = {}
    dfa_states = []
    queue = []

    start_set = frozenset([start_state])
    queue.append(start_set)
    dfa_states.append(start_set)

    while queue:
        current_set = queue.pop(0)
        dfa_transitions[frozenset(current_set)] = {}

        inputs = set(chain.from_iterable(nfa_transitions[state].keys() for state in current_set if state in nfa_transitions))
        for input_char in inputs:
            next_set = set()
            for state in current_set:
                if input_char in nfa_transitions.get(state, {}):
                    next_set.update(nfa_transitions[state][input_char])
            next_set = frozenset(next_set)
            dfa_transitions[frozenset(current_set)][input_char] = next_set
            if next_set not in dfa_states:
                dfa_states.append(next_set)
                queue.append(next_set)

    dfa_final_states = [state_set for state_set in dfa_states if any(s in final_states for s in state_set)]
    return {"transitions": dfa_transitions, "start_state": start_set, "final_states": dfa_final_states}

def generate_dfa_table(dfa):
    headers = sorted(set(chain.from_iterable(transitions.keys() for transitions in dfa["transitions"].values())))
    table = []
    for state_set, transitions_dict in dfa["transitions"].items():
        state_label = ",".join(sorted(state_set))
        row = {"State": state_label}
        for input_char in headers:
            next_state_set = transitions_dict.get(input_char, frozenset())
            next_state_label = ",".join(sorted(next_state_set))
            row[input_char] = next_state_label
        table.append(row)
    return headers, table

def generate_dfa_graph(dfa):
    try:
        G = nx.DiGraph()
        transitions = dfa["transitions"]
        start_state = dfa["start_state"]
        final_states = dfa["final_states"]

        for source_set, transitions_dict in transitions.items():
            source_label = ",".join(sorted(source_set))
            if source_set == start_state:
                color = 'lightgreen'
            elif source_set in final_states:
                color = 'salmon'
            else:
                color = 'lightblue'
            G.add_node(source_label, color=color)
            for input_char, dest_set in transitions_dict.items():
                dest_label = ",".join(sorted(dest_set))
                G.add_edge(source_label, dest_label, label=input_char)

        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42)
        node_colors = [G.nodes[node]['color'] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.9)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=30, width=2, connectionstyle='arc3,rad=0.1')
        nx.draw_networkx_labels(G, pos, font_size=14, font_weight="bold", font_color='black')
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_weight="bold", font_color='blue')
        plt.title("DFA State Transition Diagram", fontsize=20, fontweight="bold")
        plt.axis('off')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        return image_base64

    except Exception as e:
        print(f"DFA graph generation error: {e}")
        return None

@app.route("/convert-to-dfa", methods=["POST"])
def convert_to_dfa():
    data = request.json
    start_state = data.get("start_state")
    final_states = set(data.get("final_states").split(","))
    nfa_transitions = parse_nfa_transitions(data.get("transitions"))
    if not final_states:
        return jsonify({"result": "Error", "message": "Final states missing or invalid"}), 400
    # print(final_states)
    # final_states = set(final_states.split(","))

    if nfa_transitions is None:
        return jsonify({"result": "Error", "message": "Invalid NFA transition format"})
    dfa = nfa_to_dfa(nfa_transitions, start_state, final_states)
    dfa_graph = generate_dfa_graph(dfa)
    dfa_headers, dfa_table = generate_dfa_table(dfa)
    response = {
        "dfa_transitions": {str(k): {str(input_char): str(v) for input_char, v in v_dict.items()} for k, v_dict in dfa["transitions"].items()},
        "start_state": list(dfa["start_state"]),
        "final_states": [list(f) for f in dfa["final_states"]],
        "dfa_graph": dfa_graph,
        "dfa_table_headers": dfa_headers,
        "dfa_table": dfa_table
    }
    return jsonify(response)

#app2
def parse_transitions(transitions):
    """
    Parse the transition string into a structured dictionary.
    """
    transition_dict = {}
    try:
        for trans in transitions.split(";"):
            trans = trans.strip()
            if not trans:
                continue
            if "=" not in trans:
                return None
            state_input, next_state = trans.split("=", 1)

            # Split state and input, handling potential whitespace
            state_parts = state_input.split(",")
            if len(state_parts) != 2:
                return None

            state, input_char = state_parts[0].strip(), state_parts[1].strip()

            # Initialize nested dictionary if not exists
            if state not in transition_dict:
                transition_dict[state] = {}

            # Add transition
            transition_dict[state][input_char] = next_state.strip()

        return transition_dict
    except Exception:
        return None

def simulate_dfsm(num_states, initial_state, final_states, transitions, test_string):
    try:
        # Normalize final states and transitions
        final_states = [state.strip() for state in final_states.split(",")]
        transition_dict = parse_transitions(transitions)

        if transition_dict is None:
            return {
                "result": "Error",
                "message": "Invalid transition format",
                "steps": []
            }

        # Ensure all states from transitions are included in the state list
        all_states = set(transition_dict.keys())
        for state_transitions in transition_dict.values():
            all_states.update(state_transitions.values())

        # Validate final states
        for state in final_states:
            if state not in all_states:
                return {
                    "result": "Error",
                    "message": f"Final state {state} not in defined states",
                    "steps": []
                }

        current_state = initial_state
        steps = [{"state": current_state, "input": "Initial"}]

        for char in test_string:
            # Validate current state and input transition
            if current_state not in transition_dict:
                return {
                    "result": "Rejected",
                    "message": f"No transitions defined for state {current_state}",
                    "steps": steps
                }

            # If there are transitions for the state, check if the current input has a transition
            if char not in transition_dict[current_state]:
                return {
                    "result": "Rejected",
                    "message": f"No transition from state {current_state} with input {char}",
                    "steps": steps
                }

            # Move to next state
            next_state = transition_dict[current_state][char]
            steps.append({
                "state": next_state, 
                "input": char
            })
            current_state = next_state

        # Check final state
        is_accepted = current_state in final_states
        return {
            "result": "Accepted" if is_accepted else "Rejected",
            "message": f"Final state {current_state} is {'in' if is_accepted else 'not in'} final states",
            "steps": steps
        }
    except Exception as e:
        return {
            "result": "Error",
            "message": str(e),
            "steps": []
        }

def generate_dfa_graph2(initial_state, final_states, transitions):
    """
    Generate a graphical representation of the DFA.
    """
    try:
        G = nx.DiGraph()

        # Parse transitions
        transition_dict = parse_transitions(transitions)
        if transition_dict is None:
            return None

        # Normalize final states
        final_states = [state.strip() for state in final_states.split(",") if state.strip()]

        # Collect all unique states
        all_states = set(transition_dict.keys()).union(
            *[set(state_transitions.values()) for state_transitions in transition_dict.values()]
        )

        # Add all states as nodes with appropriate coloring
        for state in all_states:
            node_color = "lightblue"  # Default color
            if state == initial_state and state in final_states:
                node_color = "gold"
            elif state == initial_state:
                node_color = "lightgreen"
            elif state in final_states:
                node_color = "salmon"

            G.add_node(state, color=node_color)

        # Add edges with input symbols
        for source_state, state_transitions in transition_dict.items():
            for input_symbol, dest_state in state_transitions.items():
                G.add_edge(source_state, dest_state, label=input_symbol)

        # Create the graph layout
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)

        # Draw nodes with appropriate size and color
        node_colors = [G.nodes[node]["color"] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.9)
        nx.draw_networkx_edges(
            G, pos, edge_color="gray", arrows=True, arrowsize=20, width=2, connectionstyle="arc3,rad=0.1"
        )
        nx.draw_networkx_labels(G, pos, font_size=14, font_weight="bold", font_color="black")
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_color="blue")

        plt.title("DFA State Transition Diagram", fontsize=20, fontweight="bold")
        plt.axis("off")

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight", dpi=300)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        plt.close()

        return image_base64
    except Exception as e:
        print(f"DFA graph generation error: {e}")
        return None

@app.route("/simulate", methods=["POST"])
def simulate():
    # Get data from the form
    data = request.json
    if not all(key in data for key in ["num_states", "initial_state", "final_states", "transitions", "test_string"]):
        return jsonify({"error": "Missing required fields"}), 400

    num_states = data["num_states"]
    initial_state = data["initial_state"]
    final_states = data["final_states"]
    transitions = data["transitions"]
    test_string = data["test_string"]

    simulation_result = simulate_dfsm(num_states, initial_state, final_states, transitions, test_string)
    dfa_graph = None
    if simulation_result["result"] == "Accepted":
        dfa_graph = generate_dfa_graph2(initial_state, final_states, transitions)

    response = {
        "simulation": simulation_result,
        "dfa_graph": dfa_graph
    }
    return jsonify(response)

@app.route('/nfa2dfa')
def nfa2dfa():
    return render_template('nfa2dfa.html')

@app.route('/dfa')
def dfa():
    return render_template('dfa.html')

@app.route('/')
def index():
    return render_template('index.html')

# from flask import Flask, render_template
# app = Flask(__name__)

# @app.route('/templates/dfa.html')
# def dfa():
#     return render_template('dfa.html')

if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=8080)


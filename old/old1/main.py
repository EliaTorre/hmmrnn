import os
import numpy as np
from scripts.pipe import main
from scripts.hmm import generate_starting_probabilities, generate_transition_matrix, generate_emission_probabilities
import argparse

def run():
    states, outputs = 20, 3
    start_probabilities = generate_starting_probabilities(states)
    transition_matrix = generate_transition_matrix(states, stay_prob=0.95)
    emission_probabilities = generate_emission_probabilities(states, outputs)

    folder = ["20states_3outs_95%"]
    base_folder = folder[0]
    
    # Define the path to the base folder (ensuring a trailing separator)
    path = os.path.join(base_folder, "")
    os.makedirs(base_folder, exist_ok=True)
    
    # Create the subfolders for models and figures inside the base folder
    models_path = os.path.join(path, "models")
    figs_path = os.path.join(path, "figs")
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(figs_path, exist_ok=True)
    
    main(start_probabilities, transition_matrix, emission_probabilities, outputs, states, path)

if __name__ == "__main__":
    run()


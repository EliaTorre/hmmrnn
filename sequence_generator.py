import torch
import torch.nn.functional as F
import numpy as np
from scripts.rnn import RNN
from hmmlearn.hmm import CategoricalHMM

def generate_rnn_sequences(model_path, timesteps=1000, input_size=100, hidden_size=150, output_size=3, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    rnn = RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    rnn.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    rnn.eval()
    
    ih = rnn.rnn.weight_ih_l0.data
    hh = rnn.rnn.weight_hh_l0.data
    fc = rnn.fc.weight.data
    
    h = torch.normal(0, 1, size=(hidden_size,), device=device)
    hidden_states = []
    outputs = []
    
    with torch.no_grad():
        for t in range(timesteps):
            x = torch.normal(0, 1, size=(input_size,), device=device)
            pre_act = h @ hh.T + x @ ih.T
            h = torch.relu(pre_act)
            logits = h @ fc.T
            output = F.gumbel_softmax(logits, tau=1, hard=True, eps=1e-10, dim=-1).cpu().numpy()
            hidden_states.append(h.cpu().numpy())
            outputs.append(output)

    outputs_array = np.array(outputs)
    one_hot_outputs = np.eye(output_size)[np.argmax(outputs_array, axis=-1)]

    return one_hot_outputs


def generate_hmm_sequences(transition_matrix, emission_matrix, timesteps=1000, num_states=None, num_outputs=None):
    transition_matrix = np.array(transition_matrix)
    emission_matrix = np.array(emission_matrix)

    model = CategoricalHMM(n_components=num_states)
    model.startprob_ = np.full(num_states, 1/num_states)  # Uniform start probabilities
    model.transmat_ = transition_matrix
    model.emissionprob_ = emission_matrix

    observations, _ = model.sample(timesteps)
    observations = observations.reshape(timesteps)

    one_hot_outputs = np.eye(num_outputs)[observations]
    
    return one_hot_outputs

def fit_hmm_to_sequences(sequences, max_states=10, min_states=2, n_iter=10000, random_state=42):
    if len(sequences.shape) == 2 and sequences.shape[1] > 1:
        observations = np.argmax(sequences, axis=1).reshape(-1, 1)
    else:
        observations = sequences.reshape(-1, 1)
    
    n_states_range = range(min_states, max_states + 1)
    models = []
    aic_scores = []
    bic_scores = []
    log_likelihoods = []
    
    print(f"Fitting HMM models with {min_states} to {max_states} states...")
    
    for n_states in n_states_range:
        # Create and fit HMM model
        model = CategoricalHMM(n_components=n_states, n_iter=n_iter, random_state=random_state)
        model.fit(observations)
        log_likelihood = model.score(observations)
        
        # Calculate number of free parameters
        # startprob: n_states - 1, transmat: n_states * (n_states - 1), emissionprob: n_states * (n_outputs - 1)
        n_outputs = len(np.unique(observations))
        n_params = (n_states - 1) + n_states * (n_states - 1) + n_states * (n_outputs - 1)
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(len(observations))
        
        models.append(model)
        aic_scores.append(aic)
        bic_scores.append(bic)
        log_likelihoods.append(log_likelihood)
    
    best_idx = np.argmin(bic_scores)
    best_n_states = n_states_range[best_idx]
    best_model = models[best_idx]
    
    return aic_scores, bic_scores, log_likelihoods

def fit_hmm_fixed_states(sequences, n_states, n_iter=100000, random_state=42):
    if len(sequences.shape) == 2 and sequences.shape[1] > 1:
        observations = np.argmax(sequences, axis=1).reshape(-1, 1)
    else:
        observations = sequences.reshape(-1, 1)

    model = CategoricalHMM(n_components=n_states, n_iter=n_iter, random_state=random_state)
    model.fit(observations)
    bic = model.bic(observations)
    
    transition_probs = model.transmat_
    emission_probs = model.emissionprob_

    return transition_probs, emission_probs, bic

#def wrapper(model_path, timesteps=1000, input_size=100, hidden_size=150, output_size=3, transition_matrix=None, emission_matrix=None, num_states=None, num_outputs=None, random_state=42):
    #hmm_seq = generate_hmm_sequences(transition_matrix, emission_matrix, timesteps, num_states, num_outputs)
    #rnn_seq = generate_rnn_sequences(model_path, timesteps, input_size, hidden_size, output_size)
    #aic_hmm, bic_hmm, log_likelihoods = fit_hmm_to_sequences(hmm_seq, random_state=random_state)
    #aic_rnn, bic_rnn, log_likelihoods_rnn = fit_hmm_to_sequences(rnn_seq, random_state=random_state)
    #tr_hmm, em_hmm, _ = fit_hmm_fixed_states(hmm_seq, num_states, random_state=random_state)
    #tr_rnn, em_rnn, _ = fit_hmm_fixed_states(rnn_seq, num_states, random_state=random_state)
    #return aic_hmm, bic_hmm, log_likelihoods, aic_rnn, bic_rnn, log_likelihoods_rnn, tr_hmm, em_hmm, tr_rnn, em_rnn

def wrapper(model_path, timesteps=1000, input_size=100, hidden_size=150, output_size=3, transition_matrix=None, emission_matrix=None, num_states=None, num_outputs=None, random_state=42):
    hmm_seq = generate_hmm_sequences(transition_matrix, emission_matrix, timesteps, num_states, num_outputs)
    rnn_seq = generate_rnn_sequences(model_path, timesteps, input_size, hidden_size, output_size)
    hmm_seq = np.array(hmm_seq)
    rnn_seq = np.array(rnn_seq)
    aic_hmm, bic_hmm, log_likelihoods = fit_hmm_to_sequences(hmm_seq.copy(), random_state=np.random.randint(0, 10000))
    aic_rnn, bic_rnn, log_likelihoods_rnn = fit_hmm_to_sequences(rnn_seq.copy(), random_state=np.random.randint(0, 10000))
    tr_hmm, em_hmm, _ = fit_hmm_fixed_states(hmm_seq.copy(), num_states, random_state=random_state)
    tr_rnn, em_rnn, _ = fit_hmm_fixed_states(rnn_seq.copy(), num_states, random_state=random_state)
    return aic_hmm, bic_hmm, log_likelihoods, aic_rnn, bic_rnn, log_likelihoods_rnn, tr_hmm, em_hmm, tr_rnn, em_rnn
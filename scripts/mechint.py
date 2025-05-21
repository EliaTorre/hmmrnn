import os, torch, matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from scripts.rnn import RNN
from sklearn.decomposition import PCA
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection

def load_model(model_path, input_size=100, hidden_size=150, output_size=3):
    rnn = RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    rnn.load_model(model_path)
    return rnn

def residency_plot(title, model_path, sigma=1, max_steps=50, n_samples=20, t0=20, T=1000):
    # Load model
    rnn = RNN(input_size=100, hidden_size=150, num_layers=1, output_size=3)
    rnn.load_state_dict(torch.load(model_path))
    ih = rnn.rnn.weight_ih_l0.data
    hh = rnn.rnn.weight_hh_l0.data
    fc = rnn.fc.weight.data
    device = ih.device
    
    # Initialize hidden state and warm-up
    h = torch.zeros(150).to(device)
    for _ in range(t0):
        x = torch.normal(mean=0, std=sigma, size=(100,)).float().to(device)
        h = torch.relu(x @ ih.T + h @ hh.T)
    
    # Generate sequence of hidden states
    h_sequence = []
    for _ in range(T):
        x = torch.normal(mean=0, std=sigma, size=(100,)).float().to(device)
        h = torch.relu(x @ ih.T + h @ hh.T)
        h_sequence.append(h)
    h_sequence = torch.stack(h_sequence)

    def compute_metrics(h, ih, hh, fc, sigma, max_steps, n_samples, device):
        logits = h @ fc.T
        probs = torch.softmax(logits, dim=0)
        most_probable = torch.argmax(probs).item()
        initial_logit = logits[most_probable].item()
        
        steps_list = []
        sign_changes_list = []
        
        for _ in range(n_samples):
            h_current = h.clone()
            prev_logit = initial_logit
            sign_changes = 0
            prev_delta_sign = None
            steps = 0
            while steps < max_steps:
                x = torch.normal(mean=0, std=sigma, size=(100,)).float().to(device)
                h_current = torch.relu(x @ ih.T + h_current @ hh.T)
                new_logits = h_current @ fc.T
                new_probs = torch.softmax(new_logits, dim=0)
                new_most_probable = torch.argmax(new_probs).item()
                if new_most_probable != most_probable:
                    steps_list.append(steps + 1)
                    sign_changes_list.append(sign_changes)
                    break
                current_logit = new_logits[most_probable].item()
                delta_logit = current_logit - prev_logit
                current_sign = 1 if delta_logit > 0 else (-1 if delta_logit < 0 else 0)
                if prev_delta_sign is not None and current_sign != prev_delta_sign and current_sign != 0:
                    sign_changes += 1
                prev_delta_sign = current_sign
                prev_logit = current_logit
                steps += 1
            else:
                steps_list.append(max_steps)
                sign_changes_list.append(sign_changes)
        
        avg_steps = np.mean(steps_list)
        avg_sign_changes = np.mean(sign_changes_list)
        return avg_steps, avg_sign_changes
    
    # Compute residency time and sign changes for each hidden state
    avg_steps_list = []
    avg_sign_changes_list = []
    for h in tqdm(h_sequence, desc="Computing metrics"):
        avg_steps, avg_sign_changes = compute_metrics(h, ih, hh, fc, sigma, max_steps, n_samples, device)
        avg_steps_list.append(avg_steps)
        avg_sign_changes_list.append(avg_sign_changes)
    
    avg_steps = np.array(avg_steps_list)
    avg_sign_changes = np.array(avg_sign_changes_list)
    
    # Apply PCA
    h_np = h_sequence.cpu().numpy()
    pca = PCA(n_components=2)
    h_pca = pca.fit_transform(h_np)
    
    # Compute number of unstable eigenvalues for each hidden state
    num_unstable_list = []
    for h in h_sequence:
        x = torch.normal(mean=0, std=sigma, size=(100,)).float().to(device)
        pre_act = x @ ih.T + h @ hh.T
        D = torch.diag((pre_act > 0).float())
        J = D @ hh
        eigvals = torch.linalg.eigvals(J)
        mobius = (eigvals - 1) / (eigvals + 1 + 1e-8)
        num_unstable = (mobius.real > 0).sum().item()
        num_unstable_list.append(num_unstable)
    
    # Segment hidden states into three groups
    cutoffcol = 2
    logits = torch.softmax(h_sequence @ fc.T, dim=1)
    group1 = avg_steps < 2
    group2 = (avg_steps >= cutoffcol) & (avg_steps <= 4*cutoffcol)
    group3 = (avg_steps > 4*cutoffcol) & ((logits[:, 0].cpu().numpy() > 0.8) | (logits[:, 2].cpu().numpy() > 0.8))
    
    # Compute average number of unstable eigenvalues and residency time for each group
    avg_num_unstable_group1 = np.mean([num_unstable_list[i] for i in range(T) if group1[i]]) if np.any(group1) else 0
    avg_num_unstable_group2 = np.mean([num_unstable_list[i] for i in range(T) if group2[i]]) if np.any(group2) else 0
    avg_num_unstable_group3 = np.mean([num_unstable_list[i] for i in range(T) if group3[i]]) if np.any(group3) else 0
    avg_residency_group1 = np.mean([avg_steps_list[i] for i in range(T) if group1[i]]) if np.any(group1) else 0
    avg_residency_group2 = np.mean([avg_steps_list[i] for i in range(T) if group2[i]]) if np.any(group2) else 0
    avg_residency_group3 = np.mean([avg_steps_list[i] for i in range(T) if group3[i]]) if np.any(group3) else 0
    
    # Create subplots with four panels in 1x4 layout
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))

    # Set up custom colormap with BoundaryNorm
    cmap = matplotlib.colormaps['inferno']
    color_map = cmap(np.linspace(0, 1, 15))
    colors = [
        color_map[10],
        color_map[9],
        np.array([0.6, 0.0, 0.0, 1.0]),
        np.array([0.5, 0.5, 0.5, 1.0]),
    ]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm([1, cutoffcol, 2*cutoffcol, 4*cutoffcol+1, np.max(avg_steps)], cmap.N)
    
    # First subplot: PCA colored by residency time
    sc1 = ax1.scatter(h_pca[:, 0], h_pca[:, 1], c=avg_steps, cmap=cmap, norm=norm, s=2, alpha=0.9)
    ax1.set_xlabel('PC1', fontsize=14)
    ax1.set_ylabel('PC2', fontsize=14)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Second subplot: Histogram of sign changes colored by average residency time
    counts, bin_edges = np.histogram(avg_sign_changes, bins=int(max(avg_sign_changes)))
    bin_avg_steps = []
    for i in range(len(bin_edges) - 1):
        mask = (avg_sign_changes >= bin_edges[i]) & (avg_sign_changes < bin_edges[i+1])
        bin_avg_steps.append(np.mean(avg_steps[mask]) if np.any(mask) else 0)
    
    for i in range(len(counts)):
        norm_value = norm(bin_avg_steps[i])
        color = cmap(norm_value)
        ax2.bar((bin_edges[i] + bin_edges[i+1]) / 2, counts[i], 
                width=bin_edges[i+1] - bin_edges[i], color=color, alpha=0.9)
    ax2.set_xlabel('Avg. Sign Changes in Logits Gradient', fontsize=14)
    ax2.set_ylabel('Frequency', fontsize=14)
    
    # Third subplot: Bar plot of average unstable eigenvalues per group, colored by average residency time
    groups = ['RT < 2\n(Transition)', '2 <= RT <= 8\n(Kick-Zone)', 'RT > 8\n(Cluster)']
    avg_num_unstable = [avg_num_unstable_group1, avg_num_unstable_group2, avg_num_unstable_group3]
    avg_residency = [avg_residency_group1, avg_residency_group2, avg_residency_group3]
    bar_colors = [cmap(norm(avg_residency[i])) for i in range(len(groups))]
    ax3.bar(groups, avg_num_unstable, color=bar_colors, alpha=0.9)
    ax3.set_ylabel('Avg. # Eigenvalues Re > 0\n(after Mobius Transform)', fontsize=14)
    ax3.tick_params(axis='x', labelsize=13) 

    # Set up second custom colormap for range 0-3
    colors2 = [
        np.array([0.5, 0.5, 0.5, 1.0]),
        np.array([0.6, 0.0, 0.0, 1.0]),
        color_map[10],
    ]
    cmap2 = mcolors.ListedColormap(colors2)
    norm2 = mcolors.BoundaryNorm([0, 1, 2, 3], cmap2.N)

    # Fourth subplot: PCA colored by number of unstable eigenvalues
    sc2 = ax4.scatter(h_pca[:, 0], h_pca[:, 1], c=num_unstable_list, cmap=cmap2, norm=norm2, s=2, alpha=0.9)
    ax4.set_xlabel('PC1', fontsize=14)
    ax4.set_ylabel('PC2', fontsize=14)
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    # Adjust layout and add colorbars
    plt.subplots_adjust(bottom=0.25, wspace=0.3)
    
    # Colorbar for first subplot, spanning first three subplots
    cbar1 = fig.colorbar(sc1, ax=[ax1, ax2, ax3], orientation='horizontal', pad=0.15, aspect=60)
    cbar1.set_ticks([1, cutoffcol, 2*cutoffcol, 4*cutoffcol+1, np.max(avg_steps)])
    cbar1.ax.set_xticklabels(['1', str(cutoffcol), str(2*cutoffcol), str(4*cutoffcol+1), f'{np.max(avg_steps).astype(int)}'], fontsize=14)
    cbar1.set_label('Residency Time (RT)', fontsize=14)
    
    # Colorbar for fourth subplot
    cbar2 = fig.colorbar(sc2, ax=ax4, orientation='horizontal', pad=0.15, aspect=20)
    cbar2.set_label('# Eigenvalues Re > 0', fontsize=14)
    cbar2.set_ticks([0.5, 1.5, 2.5])
    cbar2.ax.set_xticklabels(['0', '1', '2'], fontsize=14)
    
    plt.show()

def neuron_activities(model_path, initial_hidden_states=None, specified_neurons=[83, 59, 6, 28, 72, 114], n_steps=1000, device=None):
    """Simulate RNN trajectories and plot the activities of specified neurons, given a model path."""
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model parameters
    input_size = 100
    hidden_size = 150
    num_layers = 1
    output_size = 3

    # Load the RNN model
    rnn = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device)
    rnn.load_state_dict(torch.load(model_path, map_location=device))
    rnn.eval()

    # Extract weights
    ih = rnn.rnn.weight_ih_l0.data
    hh = rnn.rnn.weight_hh_l0.data
    # fc = rnn.fc.weight.data  # Not used in simulation

    # Handle initial hidden states
    if initial_hidden_states is None:
        # Default to 1 trajectory with random initial state if none provided
        n_trajectories = 1
        initial_hidden_states = torch.normal(0, 1, size=(n_trajectories, hidden_size), device=device)
    else:
        initial_hidden_states = initial_hidden_states.to(device)
        n_trajectories = initial_hidden_states.shape[0]

    # Initialize tensors for simulation
    step_norm = torch.zeros((n_trajectories, n_steps, hidden_size), device=device)
    pre_activations = torch.zeros((n_trajectories, n_steps, len(specified_neurons)), device=device)
    step_norm[:, 0, :] = initial_hidden_states

    # Simulate the RNN for n_steps
    with torch.no_grad():
        for i in range(n_trajectories):
            for j in range(n_steps - 1):
                x = torch.normal(0, 1, size=(input_size,), device=device)
                pre_act = step_norm[i, j, :] @ hh.T + x @ ih.T
                step_norm[i, j + 1, :] = torch.relu(pre_act)
                pre_activations[i, j, :] = pre_act[specified_neurons]
            # Last step pre-activation
            x = torch.normal(0, 1, size=(input_size,), device=device)
            pre_activations[i, n_steps - 1, :] = (step_norm[i, n_steps - 1, :] @ hh.T + x @ ih.T)[specified_neurons]

    # Convert to numpy
    step_norm = step_norm.cpu().numpy()
    pre_activations = pre_activations.cpu().numpy()

    # Fit and apply PCA
    pca = PCA(n_components=2)
    step_norm_flat = step_norm.reshape(-1, hidden_size)
    pca.fit(step_norm_flat)
    step_norm_pca = pca.transform(step_norm_flat).reshape(n_trajectories, n_steps, 2)

    # Determine color range
    global_min = pre_activations.min()
    global_max = pre_activations.max()
    cmap = plt.get_cmap('seismic')

    # Create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

    # Plot trajectories for each neuron
    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            neuron_pos = row * 3 + col
            for i in range(n_trajectories):
                points = step_norm_pca[i, :, :]
                segments = np.array([points[j:j+2] for j in range(n_steps - 1)])
                colors = pre_activations[i, :-1, neuron_pos]
                lc = LineCollection(segments, cmap=cmap, alpha=0.5)
                lc.set_array(colors)
                ax.add_collection(lc)
                ax.scatter(points[:, 0], points[:, 1], c=pre_activations[i, :, neuron_pos], 
                           cmap=cmap, s=10, alpha=0.7)
            ax.set_xlabel('PC1', fontsize=14)
            ax.set_ylabel('PC2', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])

    # Add colorbar
    sm = ScalarMappable(cmap=cmap)
    sm.set_array(np.linspace(global_min, global_max, 100))
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), label='Pre-Activation Value', 
                        orientation='vertical', fraction=0.02, pad=0.1)
    cbar.set_label('Pre-Activation Value', fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    plt.show()

def relu(x):
    return np.maximum(0, x)

def forward(Wr, Wn, h0, u_vec, ret_intermediate=False):
    # Store intermediate states
    if ret_intermediate:
        h_intermediate = [h0.copy()]
    for u in u_vec:
        h0 = relu(Wr @ h0 + Wn @ u)
        if ret_intermediate:
            h_intermediate.append(h0.copy())
    if ret_intermediate:
        return h_intermediate
    return h0

def mobius_transform(z):
    return (z-1)/(z+1)

def jacobian_inj(W_r, h, W_n, n, ret_D=False, c=None, mu=1, cn=None, mun=1):
    """Compute the Jacobian with injection."""
    ct = np.ones(W_r.shape[0])
    if c is not None:
        ct[c!=0] *= c[c!=0]*mu
    ctn = np.ones(W_r.shape[0])
    if cn is not None:
        ctn[cn!=0] *= cn[cn!=0]*mun
    D = np.diag(((W_r @ h * ct + (W_n @ n) * ctn) > 0).astype(float))
    if ret_D:
        return D @ W_r * ct, D
    return D @ W_r * ct

def forward_inj(Wr, Wn, h0, u_vec, ret_intermediate=False, c=None, mu=1, cn=None, mun=1):
    """Forward pass with injection."""
    ct = np.ones(Wr.shape[0])
    if c is not None:
        ct[c!=0] *= c[c!=0]*mu
    ctn = np.ones(Wr.shape[0])
    if cn is not None:
        ctn[cn!=0] *= cn[cn!=0]*mun
    if ret_intermediate:
        h_intermediate = [h0.copy()]
    for u in u_vec:
        h0 = relu((Wr @ h0)*ct + (Wn @ u)*ctn)
        if ret_intermediate:
            h_intermediate.append(h0.copy())
    if ret_intermediate:
        return h_intermediate
    return h0

def kprojected(W_r, id, top_k=10, reversed=True):
    """Project top-k neurons based on weights."""
    mask = np.zeros(W_r.shape[0]).astype(bool)
    idx = W_r[id, :].argsort()
    if reversed:
        idx = idx[::-1]
    mask[idx[:top_k]] = True
    return mask

def run_injected(W_r, W_n, W_o, tc, h_nominal, noise_nominal, trajs=100, trajlen=100, muvals=[0, 1, 2], c=None, cn=None, T=1000, sigmaj=1, gamm_noise=0.5):
    """Run simulations with injected interventions."""
    h0j = (np.random.randn(W_r.shape[0]) * 2 - 1) * 0.1
    noise_vec_j = np.random.normal(0, sigmaj, [T, W_n.shape[1]])
    noise_traj = np.random.normal(0, sigmaj, [trajs, trajlen, W_n.shape[1]])
    spwn = tc - 1
    
    mu_data = {}
    for i, mu in enumerate(muvals):
        h_istj_mu = [h0j.copy()]
        hj_mu = h_istj_mu[0]
        oj_mu = []
        J_mu_l = []
        traj_l = []
        
        for t in range(T):
            noisej = noise_vec_j[t]
            hj_mu = forward_inj(W_r, W_n, hj_mu, [noisej], c=c, mu=mu, cn=cn, mun=mu)
            h_istj_mu.append(hj_mu)
            oj_mu.append((W_o @ hj_mu).argmax())
            Jt_, _ = jacobian_inj(W_r, hj_mu, W_n, noisej, ret_D=True, c=c, mu=mu, cn=cn, mun=mu)
            J_mu_l.append(Jt_)
        
        for j in range(trajs):
            h0t = h_nominal[spwn].copy()
            traj_vec = [h0t.copy()]
            for t in range(trajlen):
                noisej = noise_nominal[spwn + t] * (1 - gamm_noise) + gamm_noise * noise_traj[j, t]
                h0t = forward_inj(W_r, W_n, h0t, [noisej], c=c, mu=mu, cn=cn, mun=mu)
                traj_vec.append(h0t)
            traj_l.append(np.array(traj_vec))
        
        mu_data[mu] = {
            'h_istj_mu': np.array(h_istj_mu),
            'oj_mu': np.array(oj_mu),
            'traj_l': np.array(traj_l),
            'J_mu_l': J_mu_l
        }
    return mu_data

def compute_critical_eigs(mu_data, muvals, dt_transit=15, do_mobius=True, eps_real=0.1, eps_img=0.06):
    """Compute mean and std of critical complex eigenvalues."""
    critical_complex_eigs_mean = {}
    critical_complex_eigs_std = {}
    for muv in muvals:
        a = np.where(np.abs(np.diff(mu_data[muv]["oj_mu"]) > 0))[0]
        if len(a) < 2:
            a = np.where(np.abs(np.diff(mu_data[1]["oj_mu"]) > 0))[0]
        comlstats = []
        for k in range(len(a) - 1):
            efdttr = dt_transit
            eigenvalues = []
            start_idx = max(0, a[k] - efdttr)
            end_idx = min(len(mu_data[muv]["J_mu_l"]), a[k] + efdttr)
            for J in mu_data[muv]["J_mu_l"][start_idx:end_idx]:
                eigvals = np.linalg.eigvals(J)
                eigenvalues.extend(eigvals)
            eigenvalues = np.array(eigenvalues)
            if do_mobius:
                eigenvalues = (eigenvalues - 1) / (eigenvalues + 1)
            comlstats.append(np.logical_and(np.abs(eigenvalues.real) < eps_real, eigenvalues.imag > eps_img).sum())
        critical_complex_eigs_mean[muv] = np.mean(comlstats) if comlstats else 0
        critical_complex_eigs_std[muv] = np.std(comlstats) if comlstats else 0
    return critical_complex_eigs_mean, critical_complex_eigs_std


def load_weights(path):
    """Load weights from a file."""
    weights = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
    W_r = weights["rnn.weight_hh_l0"].cpu().numpy() 
    W_n = weights["rnn.weight_ih_l0"].cpu().numpy()
    W_o = weights["fc.weight"].cpu().numpy()
    return W_r, W_n, W_o

def find_transition(W_r, W_n, W_o, sigma=1, back=25, skipped_trasit=4, t0=20, tf=5, trials=20, dT=20, Tmin=200, T=2000, tau=0.5):
    hj = (np.random.randn(W_r.shape[0])*2-1)*0.1
    for t in range(t0):
        noisej = np.random.normal(0, sigma, W_n.shape[1])  # noise
        hj = forward(W_r, W_n, hj, [noisej])  # Forward pass
    
    skips = 0
    ocurr = (W_o @ hj).argmax()
    h_track = [hj.copy()]
    o_track = [ocurr.copy()]
    noise_vec = np.random.normal(0, sigma, [T, W_n.shape[1]])
    pskip=0
    for t in range(T):
        noisej = noise_vec[t] # noise
        hj = forward(W_r, W_n, hj, [noisej])  # Forward pass
        h_track.append(hj)
        oj = (W_o @ hj).argmax()  # Output
        o_track.append(oj.copy())
    
    noise_vec = np.concatenate([np.zeros([1, W_n.shape[1]]), noise_vec], axis=0)
    h_track = np.array(h_track)
    o_track = np.array(o_track)

    skip = skipped_trasit
    it = max(np.diff(np.where(np.diff(o_track[Tmin:])!=0)[0][skip:-1]).argmax()-1, 0)

    tt = np.where(np.diff(o_track[Tmin:])!=0)[0][skip:-1][it+1] + Tmin
    tc = int(np.where(np.diff(o_track[Tmin:])!=0)[0][skip:-1][it+1]*tau + np.where(np.diff(o_track[Tmin:])!=0)[0][skip:-1][it]*(1-tau)) + Tmin
    tc = max(tc, tt-back)

    firing_t = []
    firing_trials_rate = []
    h_track2 = []
    o_track2 = []
    for i in range((tt-tc)+tf+1):
        trial_data = []
        fired = 0
        for j in range(trials):
            ocurr = o_track[tf+tt-i].copy()
            hj = h_track[tf+tt-i].copy()
            for t in range(dT):
                noisej = np.random.normal(0, sigma, W_n.shape[1])
                hj = forward(W_r, W_n, hj, [noisej])
                oj = (W_o @ hj).argmax()
                if ocurr != oj:
                    fired += 1
                    break
            trial_data.append(t+1)
        firing_t.append(np.mean(trial_data))  
        firing_trials_rate.append(fired/trials)

    firing_t = np.array(firing_t)
    firing_trials_rate = np.array(firing_trials_rate)

    return firing_t, firing_trials_rate, h_track, o_track, noise_vec, tt, tc

def locate_areas(W_r, W_n, W_o, T=2000, thr_fr=6, verbose=False, max_retries=5):
    """Locate transition and cluster points"""
    sigma = 1
    t0 = 20
    tau = 0.1
    back = 20
    tf = 5
    trials = 100
    skip = 4
    Tmin = 200
    dT = max(thr_fr, 25)

    for attempt in range(max_retries):
        # Initialize state
        hj = (np.random.randn(W_r.shape[0]) * 2 - 1) * 0.1
        for t in range(t0):
            noisej = np.random.normal(0, sigma, W_n.shape[1])
            hj = forward_inj(W_r, W_n, hj, [noisej])

        # Run simulation
        h_track = [hj.copy()]
        o_track = [(W_o @ hj).argmax()]
        noise_vec = np.random.normal(0, sigma, [T, W_n.shape[1]])
        for t in range(T):
            noisej = noise_vec[t]
            hj = forward_inj(W_r, W_n, hj, [noisej])
            h_track.append(hj)
            o_track.append((W_o @ hj).argmax())

        noise_vec = np.concatenate([np.zeros([1, W_n.shape[1]]), noise_vec], axis=0)
        h_track = np.array(h_track)
        o_track = np.array(o_track)

        # Find transition point (tt)
        diff_indices = np.where(np.diff(o_track[Tmin:]) != 0)[0]
        if len(diff_indices[skip:-1]) <= 0:
            if verbose:
                print(f"Attempt {attempt + 1}/{max_retries}: No valid transitions found after Tmin. Setting it = 0, tt = Tmin.")
            it = 0
            tt = Tmin
        else:
            it = max(np.diff(diff_indices[skip:-1]).argmax() - 1, 0)
            tt = diff_indices[skip:-1][it + 1] + Tmin

        # Estimate cluster point (tc)
        if len(diff_indices[skip:-1]) <= it + 1:
            if verbose:
                print(f"Attempt {attempt + 1}/{max_retries}: Insufficient transitions for tc. Setting tc = tt - back.")
            tc = tt - back
        else:
            tc = max(int(diff_indices[skip:-1][it + 1] * tau + 
                        diff_indices[skip:-1][it] * (1 - tau)) + Tmin, tt - back)

        # Compute firing times and rates
        firing_t = []
        firing_trials_rate = []
        for i in range((tt - tc) + tf + 1):
            trial_data = []
            fired = 0
            for j in range(trials):
                hj = h_track[tf + tt - i].copy()
                ocurr = o_track[tf + tt - i]
                for t in range(dT):
                    noisej = np.random.normal(0, sigma, W_n.shape[1])
                    hj = forward_inj(W_r, W_n, hj, [noisej])
                    oj = (W_o @ hj).argmax()
                    if ocurr != oj:
                        fired += 1
                        break
                trial_data.append(t + 1)
            firing_t.append(np.mean(trial_data))
            firing_trials_rate.append(fired / trials)

        firing_t = np.array(firing_t)
        firing_trials_rate = np.array(firing_trials_rate)

        # Compute kick point (tk)
        condition_indices = np.where((firing_t < 3) & (firing_trials_rate == 1))[0]
        if len(condition_indices) > 0:
            tk = tt - (condition_indices[-1] - tf)
        else:
            tk = tt
            if verbose:
                print(f"Attempt {attempt + 1}/{max_retries}: No points satisfy (firing_t < 3) & (firing_trials_rate == 1). Setting tk = tt.")

        # Check cluster mask
        cluster_mask = (firing_t >= thr_fr) & (firing_trials_rate < 1) & (np.arange(len(firing_t)) > tf)
        if np.sum(cluster_mask) > 0:
            tc = tt - (np.where(cluster_mask)[0][0] - tf)
            if verbose:
                print(f"Attempt {attempt + 1}/{max_retries}: Valid cluster_mask found. Exiting retry loop.")
            break
        if verbose:
            print(f"[...] REITERATING (Attempt {attempt + 1}/{max_retries})")
    else:
        tc = max(tc, tt - back)
        if verbose:
            print(f"Warning: Max retries ({max_retries}) reached. Retaining tc = max(tc, tt - back).")

    return h_track, o_track, noise_vec, tt, tk, tc


def noise_sensitivity(title, W_r, W_n, o_track, h_track, noise_vec, tk, tc, gamm_noise_vals=[0,0.5,1], T=6, trajs=100, beta_fac=0):
    sigmaj = 1
    
    noise_vec = np.concatenate([noise_vec, np.random.normal(0, sigmaj, [T, noise_vec.shape[1]])], axis=0)  # noise

    idx_vals = [tk, tc] 
    gamm_noise_vals = gamm_noise_vals
    pca_hj = PCA(n_components=2)
    pca_hj.fit(h_track)
    h_track_pca = pca_hj.transform(h_track)

    trans_trajs = {}
    trans_trajs_pca = {}
    for idx in idx_vals:
        gamm_jacobs = {}
        gamm_trajs = {}
        gamm_trajs_pca = {}
        h0_ch = h_track[idx].copy()
        for gamm_noise in gamm_noise_vals:
            ch_trajs = []
            for _ in range(trajs):
                hj = h0_ch.copy() + np.random.normal(0, beta_fac, h0_ch.shape[0])  # Random initial hidden state
                traj_k = [hj.copy()]
                for t in range(T):
                    noisej = noise_vec[idx+t]*(1-gamm_noise) + gamm_noise*np.random.normal(0, sigmaj, noise_vec.shape[1]) #* (np.random.rand()*1.5 if t==0 else 1)  #+ np.random.normal(0, 0.001, W_n.shape[1]).flatten() if t==0 else np.zeros_like(noise_vec[t])  # noise
                    hj = forward(W_r, W_n, hj, [noisej])  # Forward pass
                    traj_k.append(hj)

                ch_trajs.append(np.array(traj_k))
            ch_trajs = np.array(ch_trajs)

            ch_trajs_pca = pca_hj.transform(ch_trajs.reshape(-1, h0_ch.shape[0])).reshape(trajs, T+1, 2)
            gamm_trajs[gamm_noise] = ch_trajs
            gamm_trajs_pca[gamm_noise] = ch_trajs_pca
        trans_trajs[idx] = gamm_trajs
        trans_trajs_pca[idx] = gamm_trajs_pca

    #### PLOT 1

    # Define custom colormap for o_track (0: dark green, 2: dark red)
    colors = {0: 'dimgrey', 1:"royalblue", 2: 'darkgrey'}
    cmap = mcolors.ListedColormap([colors[0], colors[2]])
    max_trajs = 30 # trajs
    start = 1

    rows = 1  # One row for all subplots
    columns = len(gamm_noise_vals) * len(idx_vals)  # One column per gamma value per index
    fig, axs = plt.subplots(rows, columns, figsize=(24, 4))  # 1x6 subplot grid

    for i, idj in enumerate(idx_vals):
        for j, gni in enumerate(gamm_noise_vals):
            col_idx = i * len(gamm_noise_vals) + j
            # Set title based on cluster or transition
            title2 = f"Cluster ($\gamma=${gni})" if idj == idx_vals[0] else f"Transition ($\gamma=${gni})"
            axs[col_idx].set_title(title2, fontsize=16)
            axs[col_idx].scatter(h_track_pca[:, 0], h_track_pca[:, 1], c=o_track, cmap=cmap, alpha=0.3)
            for k in range(max_trajs):
                axs[col_idx].plot(trans_trajs_pca[idj][gni][k, ..., 0], trans_trajs_pca[idj][gni][k, ..., 1], color="darkblue", alpha=0.6)
            axs[col_idx].scatter(trans_trajs_pca[idj][gni][:,0,0], trans_trajs_pca[idj][gni][:,0,1], c='black', alpha=1, marker="*", s=100)
            if col_idx == 0:
                axs[col_idx].set_ylabel("PC2", fontsize=16)
            axs[col_idx].set_xlabel("PC1", fontsize=16)
            axs[col_idx].set_xticks([])
            axs[col_idx].set_yticks([])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    #### PLOT 2
    start = 0
    n, T, D = trans_trajs[idj][gni].shape
    X = trans_trajs[idj][gni][:, start:, :]
    plt.figure(figsize=(8, 5))
    plt.suptitle(title, fontsize=14)
    noise_sensitivity_dt = {}
    # Compute spread metrics over time
    for idj in idx_vals:
        final_cov_trace = []
        final_mean_distance = []
        for gk, gni in enumerate(gamm_noise_vals):
            if gni == 0: continue
            traces = []
            mean_distances = []
            hull_areas = []
            for t in range(T):
                X_t = trans_trajs[idj][gni][:, t, :]
                mean_X = np.mean(X_t, axis=0)
                centered_X = X_t - mean_X
                # Trace of covariance
                trace = np.sum((centered_X ** 2) / (n - 1))
                traces.append(trace)
                # Mean distance to mean
                distances = np.sqrt(np.sum(centered_X ** 2, axis=1))
                mean_distances.append(np.mean(distances))
                
            # Determine label and color based on whether idj is cluster or transition
            # and use different shades based on gamma value
            if idj == idx_vals[0]:  # Cluster
                label = f"Cluster (γ={gni})"
                # Use different shades of blue based on gamma index
                if gk == 0 or (gk == 1 and gamm_noise_vals[0] == 0):
                    color = "royalblue"  # Lighter blue for first non-zero gamma
                else:
                    color = "darkblue"   # Darker blue for second gamma
            else:  # Transition
                label = f"Transition (γ={gni})"
                # Use different shades of red based on gamma index
                if gk == 0 or (gk == 1 and gamm_noise_vals[0] == 0):
                    color = "indianred"  # Lighter red for first non-zero gamma
                else:
                    color = "darkred"    # Darker red for second gamma
                
            # Plot metrics with specific colors
            plt.subplot(1, 2, 1)
            plt.plot(traces, label=label, color=color)
            plt.title("Trace of Covariance", fontsize=14)
            plt.xlabel("Time", fontsize=14)
            plt.ylabel("Trace", fontsize=14)
            
            plt.subplot(1, 2, 2)
            plt.plot(mean_distances, label=label, color=color)
            plt.title("Avg. Distance to Mean Trajectory", fontsize=14)
            plt.xlabel("Time", fontsize=14)
            plt.ylabel("Mean Distance", fontsize=14)
            
            plt.tight_layout()
            final_cov_trace.append(np.mean(traces[-5:]))
            final_mean_distance.append(np.mean(mean_distances[-5:]))
            
        noise_sensitivity_dt[idj] = {
            "final_cov_trace": final_cov_trace,
            "final_mean_distance": final_mean_distance
        }

    plt.legend(fontsize=10)
    plt.show()

def weight_matrices(W_r):
    """Plot weight matrices for kick neurons and integrating populations."""
    # Create figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    
    # Define neuron indices for kick groups
    vec = np.array([83, 6, 59, 114, 72, 28])
    
    # Plot 1: Kick Neurons ⇄ Kick Neurons
    axs[0].imshow(W_r[vec,:][:, vec], cmap='bwr', aspect='auto')
    axs[0].set_xticks(ticks=[1, 4], labels=["Kick group 1", "Kick group 2"], fontsize=14)
    axs[0].set_yticks(ticks=[1, 4], labels=["Kick group 1", "Kick group 2"], rotation=90, fontsize=14)
    axs[0].set_title("$W_{hh}$ weights: Kick Neurons ⇄ Kick Neurons", fontsize=13)
    
    # Calculate top-k integrating neurons
    topk = 70
    g1 = W_r[vec[:3],:].mean(axis=0)
    g2 = W_r[vec[3:],:].mean(axis=0)

    sortmap = np.argsort(g1*g2)

    g1 = g1[sortmap][:topk]
    g2 = g2[sortmap][:topk]
    sortmap2 = np.argsort(g1)
    final_map = sortmap[sortmap2[sortmap2<topk]]

    
    # Plot 2: Kick Neurons ⇄ Integrating Populations
    s1 = axs[1].imshow(W_r[vec, :][:, final_map], cmap='bwr', aspect='auto')
    axs[1].set_xticks(ticks=[int(topk * 0.25), int(topk * 0.75)], 
                     labels=["Integrating\npopulation 1", "Integrating\npopulation 2"], fontsize=14)
    axs[1].set_yticks(ticks=[1, 4], labels=["Kick group 1", "Kick group 2"], rotation=90, fontsize=14)
    axs[1].set_title("$W_{hh}$ weights: Kick Neurons ⇄ Integrating Populations", fontsize=13)
    
    # Plot 3: Integrating Populations ⇄ Integrating Populations
    axs[2].imshow(W_r[final_map, :][:, final_map], cmap='bwr', aspect='auto')
    axs[2].set_xticks(ticks=[int(topk * 0.25), int(topk * 0.75)], 
                     labels=["Integrating\npopulation 1", "Integrating\npopulation 2"], fontsize=14)
    axs[2].set_yticks(ticks=[int(topk * 0.25), int(topk * 0.75)], 
                     labels=["Integrating\npopulation 1", "Integrating\npopulation 2"], rotation=90, fontsize=14)
    axs[2].set_title("$W_{hh}$ weights: Integrating Pop. ⇄ Integrating Pop.", fontsize=13)
    
    # Add horizontal colorbar below all plots
    cbar = fig.colorbar(s1, ax=axs, orientation='horizontal', fraction=0.15, pad=0.05)
    cbar.set_label("Weight value")
    plt.show()

def mean_activities(model_path, T=300, kick_group1=[83,6,59], kick_group2=[114,72,28], topk=70, sigmaj=1, TMaxx=200):
    """Plot the mean activities of kick neurons and integrating populations over time."""
    # Load weights
    W_r, W_n, W_o = load_weights(model_path)

    # Set up matplotlib
    cmap = matplotlib.colormaps['inferno']
    color_map = cmap(np.linspace(0, 1, 15))
    colors = [
        color_map[10],
        color_map[9],
        color_map[7],
        color_map[1]
    ]
    cmap = mcolors.ListedColormap(colors)
    font = {'size': 16}
    matplotlib.rc('font', **font)

    # Compute integrating populations
    g1 = W_r[kick_group1, :].mean(axis=0)
    g2 = W_r[kick_group2, :].mean(axis=0)
    sortmap = np.argsort(g1 * g2)
    g1_top = g1[sortmap][:topk]
    g2_top = g2[sortmap][:topk]
    sortmap2 = np.argsort(g1_top)
    final_map = sortmap[sortmap2[sortmap2 < topk]]

    # Simulate RNN
    noise_vec_j = np.random.normal(0, sigmaj, [T, W_n.shape[1]])
    h0j = (np.random.randn(W_r.shape[0]) * 2 - 1) * 0.1
    h_track = []
    o_track = []
    for t in range(T):
        h0j = forward_inj(W_r, W_n, h0j, [noise_vec_j[t]])
        h_track.append(h0j.copy())
        o = W_o @ h0j
        o_track.append(o.argmax())
    h_track = np.array(h_track)
    o_track = np.array(o_track)

    # Filter h_track for integrating populations
    filtered = h_track[:, final_map]

    # Define masks for kick groups
    map_k1 = np.zeros(h_track.shape[1], dtype=bool)
    map_k1[kick_group2] = True  # Neurons labeled "Kick Group 1" in plot
    map_k2 = np.zeros(h_track.shape[1], dtype=bool)
    map_k2[kick_group1] = True  # Neurons labeled "Kick Group 2" in plot

    # Plot
    plt.figure(figsize=(20, 3))
    plt.plot(h_track[:TMaxx, map_k1].mean(axis=-1), label="Kick Group 1", c="navy")
    plt.plot(h_track[:TMaxx, map_k2].mean(axis=-1), label="Kick Group 2", c="darkred")
    half = filtered.shape[1] // 2
    plt.plot(filtered[:TMaxx, :half].mean(axis=-1), label="Integrating Pop 1", alpha=1, linestyle='--', c="blue")
    plt.plot(filtered[:TMaxx, half:].mean(axis=-1), label="Integrating Pop 2", alpha=1, linestyle='--', c="red")
    plt.legend(loc='upper right', fontsize=12)

    barss = np.where(np.diff(o_track[:TMaxx]) != 0)[0]
    alpha_l = 0.4
    mx = np.max([h_track[:TMaxx, map_k1].mean(axis=-1), h_track[:TMaxx, map_k2].mean(axis=-1)]) + 0.3

    if len(barss) > 0:
        for i, b in enumerate(barss):
            if i == 0:
                plt.fill_betweenx([0, mx], 0, b + 1, color='lightgray', alpha=alpha_l)
            else:
                color = 'lightgray' if i % 2 == 0 else 'darkgray'
                plt.fill_betweenx([0, mx], barss[i - 1] + 1, b + 1, color=color, alpha=alpha_l)
        last_bar = barss[-1] + 1
        last_color = 'lightgray' if len(barss) % 2 == 1 else 'darkgray'
    else:
        last_bar = 0
        last_color = 'lightgray'

    plt.fill_betweenx([0, mx], last_bar, min(TMaxx, h_track.shape[0]), color=last_color, alpha=alpha_l)
    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Mean activity", fontsize=16)
    plt.title("Mean activity of Kick Neurons and Noise Integrating Populations", fontsize=20)
    plt.show()

def pca_evolution(folder_path, selected_epochs_list, T=5000, t0=10, sigmaj=1, h_eps=0.1, maxp=700, trlen=700, clip_value=15, purple_epoch=None):
    """Generate a 3D PCA plot showing state space evolution for specified epochs."""
    # Get and sort model files
    model_files = sorted(
        [f for f in os.listdir(folder_path) if f.startswith('model_epoch_') and f.endswith('.pth')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    
    all_epochs = [int(fname.split('_')[-1].split('.')[0]) for fname in model_files]
    epochs_to_plot = selected_epochs_list
    
    # Add purple_epoch if specified and not already included
    if purple_epoch is not None and purple_epoch not in epochs_to_plot and purple_epoch in all_epochs:
        epochs_to_plot = epochs_to_plot + [purple_epoch]
    
    # Find indices for epochs_to_plot
    selected_indices = []
    selected_epochs = []
    for epoch in epochs_to_plot:
        if epoch in all_epochs:
            idx = all_epochs.index(epoch)
            selected_indices.append(idx)
            selected_epochs.append(epoch)
        else:
            print(f"Warning: Epoch {epoch} not found in model files.")
    
    # Sort by epoch number
    if selected_epochs:
        sorted_pairs = sorted(zip(selected_indices, selected_epochs), key=lambda x: x[1])
        selected_indices, selected_epochs = zip(*sorted_pairs)
        selected_indices = list(selected_indices)
        selected_epochs = list(selected_epochs)
    else:
        print("No epochs to plot.")
        return
    
    # Load first model to determine dimensions
    W_r, W_n, W_o = load_weights(os.path.join(folder_path, model_files[0]))
    hidden_size = W_r.shape[0]
    input_size = W_n.shape[1]
    
    # Initialize noise and initial hidden state
    noise_warmup = np.random.normal(0, sigmaj, [t0, input_size])
    noise_vec_j = np.random.normal(0, sigmaj, [T, input_size])
    h0j = (np.random.randn(hidden_size) * 2 - 1) * h_eps
    
    # Lists to store PCA points and outputs
    pca_points_2d = []
    out_t = []
    
    # Process each selected epoch
    for idx in selected_indices:
        model_file = os.path.join(folder_path, model_files[idx])
        W_r, W_n, W_o = load_weights(model_file)
        
        # Warmup phase
        h_current = h0j.copy()
        for t in range(t0):
            noise = noise_warmup[t]
            h_current = forward(W_r, W_n, h_current, [noise])
        
        # Simulation phase
        hj_mu = [h_current]
        for t in range(T):
            noise = noise_vec_j[t]
            h_current = forward(W_r, W_n, h_current, [noise])
            hj_mu.append(h_current)
        hj_mu = np.array(hj_mu)
        
        # Compute outputs from logits
        logits = np.einsum("ij, tj -> ti", W_o, hj_mu)
        out = np.argmax(logits, axis=1)
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca.fit(hj_mu)
        hj_mu_pca = pca.transform(hj_mu)
        hj_mu_pca = np.clip(hj_mu_pca, -clip_value, clip_value)
        
        pca_points_2d.append(hj_mu_pca)
        out_t.append(out)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=14, azim=80, roll=0)
    
    # Define fixed colors
    color_red = 'darkred'
    color_green = 'darkgreen'
    
    # Plot points and trajectories for each epoch
    for t, (cloud, out) in enumerate(zip(pca_points_2d, out_t)):
        n_points = min(maxp, cloud.shape[0])
        x = cloud[:n_points, 0]
        y = cloud[:n_points, 1]
        z = np.full(n_points, t)
        
        # Check if this is the purple epoch
        is_purple = selected_epochs[t] == purple_epoch if purple_epoch is not None else False
        
        # Color for scatter points
        scatter_color_red = 'purple' if is_purple else color_red
        scatter_color_green = 'purple' if is_purple else color_green
        
        # Plot scatter points
        mask = out[:n_points] == 2
        ax.scatter(x[mask], y[mask], z[mask], c=[scatter_color_red], s=10, alpha=0.3)
        ax.scatter(x[~mask], y[~mask], z[~mask], c=[scatter_color_green], s=10, alpha=0.3)
        
        # Plot trajectory
        if is_purple:
            # Plot entire trajectory in purple
            ax.plot(x[:trlen], y[:trlen], z[:trlen], color='purple', alpha=0.3)
        else:
            # Plot each segment based on the class of the starting point
            for i in range(trlen - 1):
                segment_color = color_red if out[i] == 2 else color_green
                ax.plot(x[i:i+2], y[i:i+2], [t, t], color=segment_color, alpha=0.3)
    
    # Set labels and ticks
    ax.set_zlabel('Epoch Index')
    ax.set_zticks(range(len(selected_epochs)))
    ax.set_zticklabels(selected_epochs)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title('PCA of Latent Dynamics Across Epochs')
    plt.show()

def expected_deltah(W_r, W_n, h0=None, trials=100, t0=10, tf=50, sigma=1, h_eps=0.1, silent=False):
    """Compute the expected first-order and second-order perturbation terms for an RNN with ReLU activation."""

    def compute_delta_h2(W_r, W_n, h0, sigma, t, Mtk_corr=False):
        """Compute the second-order perturbation term delta_h_t^{(2)} for an RNN with ReLU activation."""
        # Dimensions
        n = W_r.shape[0]  # Hidden state dimension
        r = W_n.shape[1]  # Noise dimension
        
        # Step 1: Compute the unperturbed trajectory
        h_unperturbed = [h0.copy()]
        for k in range(1, t):
            z_k = W_r @ h_unperturbed[-1]  # Pre-activation
            h_k = relu(z_k)                # Post-activation
            h_unperturbed.append(h_k)
        
        # Step 2: Compute Jacobians for the unperturbed trajectory
        # For ReLU, Jacobian is diagonal with 1 where z_k > 0, 0 otherwise
        D_phi = [np.diag((W_r @ h > 0).astype(float)) for h in h_unperturbed]
        
        # Step 3: Generate noise
        omega = np.random.normal(0, sigma, (t, r))
        
        # Step 4: Compute first-order perturbations
        delta_h1 = np.zeros((t, n))
        Mtk_l = []
        for s in range(t):
            # Compute propagation matrix M_{t,s}
            M_t_s = np.eye(n)
            for k in range(s, t):
                M_t_s = D_phi[k] @ W_r @ M_t_s
            if Mtk_corr:
                Mtk_l.append(M_t_s)
            delta_h1[s] = M_t_s @ W_n @ omega[s]
        
        # Step 5: Compute second-order term
        delta_h2 = np.zeros(n)
        for u in range(t):
            z_u = W_r @ h_unperturbed[u]  # Pre-activation at step u
            dhu = delta_h1[u]
            dv2 = (dhu)**2
            if Mtk_corr:
                Mtu = Mtk_l[u]
                dv2 = (-Mtu) @ dv2
            delta_h2 += 0.5* np.einsum("i,i->i", dv2, (z_u < 0))

        return delta_h2, delta_h1[0]

    # PERTURBATION VECS
    delta_h1_l = []
    delta_h2_l = []
    for ex in range(trials):
        if not silent:
            print(f"{ex/trials*100:.2f}%    ", end="\r")
        if h0 is None:
            h0 = (np.random.randn(W_r.shape[0])*2-1)*h_eps
        h0j = h0+(np.random.randn(W_r.shape[0])*2-1)*h_eps 
        for t in range(1, t0):
            noise = np.random.normal(0, sigma, W_n.shape[1])
            h0j = forward(W_r, W_n, h0j, [noise]) 

        # Compute the second-order noise vector
        delta_h2, delta_h1 = compute_delta_h2(W_r, W_n, h0j, sigma, tf)
        delta_h1_l.append(delta_h1)
        delta_h2_l.append(delta_h2)

    return np.array(delta_h2_l).mean(axis=0), np.array(delta_h1_l).mean(axis=0)
    #return np.array(delta_h2_l), np.array(delta_h1_l)

def second_order(folder_path, max_epochs=250, ep_rate=10, threshold=0.05, bifep=None, silent=True):
    
    def Hopf_eigs_stat(Jl, do_mobius=True, t0=0, eps_real=0.1, eps_img=0.05):
        compl_stat = []
        unstable_stat = []
        for Jt in Jl[t0:]:
            eigvals = np.linalg.eigvals(Jt)
            unstable_stat += [(np.linalg.norm(np.stack([np.real(eigvals), np.imag(eigvals)], axis=-1), axis=-1)>1).sum()]    # counts just 1 per complex pair
            if do_mobius:
                # Apply Mobius transformation to eigenvalues
                eigvals = mobius_transform(eigvals)
            compl_stat += [np.logical_and(np.abs(eigvals.real)<eps_real, eigvals.imag>eps_img).sum()]    # counts just 1 per complex pair     
        return np.max(compl_stat), np.mean(compl_stat), np.std(compl_stat), np.max(unstable_stat), np.mean(unstable_stat), np.std(unstable_stat)

    def mobius_transform(z):
        return (z-1)/(z+1)
    
    # Get and sort model files
    model_files = sorted(
        [f for f in os.listdir(folder_path) if f.startswith('model_epoch_') and f.endswith('.pth')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    
    all_epochs = [int(fname.split('_')[-1].split('.')[0]) for fname in model_files]
    epochs_to_plot = [i for i in range(1, max_epochs+1, ep_rate)]
    
    # Find indices for epochs_to_plot
    selected_indices = []
    selected_epochs = []
    for epoch in epochs_to_plot:
        if epoch in all_epochs:
            idx = all_epochs.index(epoch)
            selected_indices.append(idx)
            selected_epochs.append(epoch)
        else:
            print(f"Warning: Epoch {epoch} not found in model files.")
    
    # Sort by epoch number
    if selected_epochs:
        sorted_pairs = sorted(zip(selected_indices, selected_epochs), key=lambda x: x[1])
        selected_indices, selected_epochs = zip(*sorted_pairs)
        selected_indices = list(selected_indices)
        selected_epochs = list(selected_epochs)
    else:
        print("No epochs to plot.")
        return
    
    # Load first model to determine dimensions
    W_r, W_n, W_o = load_weights(os.path.join(folder_path, model_files[0]))
    state_dim = W_r.shape[0]
    noise_dim = W_n.shape[1]

    T = 5000
    t0 = 10
    tdh2 = 10 
    trials = 20 #100
    h_eps = .05
    sigmaj = 1
    stepJac = False

    noise_warmup = np.random.normal(0, sigmaj, [t0, noise_dim])
    noise_vec_j = np.random.normal(0, sigmaj, [T, noise_dim])
    h0j = (np.random.randn(state_dim)*2-1)*0.1 

    # Define custom colormap for o_track (0: dark green, 2: dark red)
    colors = {0: 'darkgreen', 1:"royalblue", 2: 'darkred'}
    cmap = mcolors.ListedColormap([colors[0], colors[1], colors[2]])

    vec_dh2_l = []
    vec_dh1_l = []
    compl_stats_max = []
    compl_stats_mean = []
    compl_stats_std = []
    unst_stats_max = []
    unst_stats_mean = []
    unst_stats_std = []
    avg_res_time = []
    avg_firing_rate = []
    pca_comps = []
    pca_points = []
    out_t = []
    for k, idx in enumerate(selected_indices):
        model_file = os.path.join(folder_path, model_files[idx])
        W_r, W_n, W_o = load_weights(model_file)

        if not silent:
            print(f"Epoch {idx}  -  [{k/len(selected_indices)*100:.2f}%]    ", end="\r")  
            
        # WARMUP
        for t in range(t0):
            noisej = noise_warmup[t]
            h0j = forward(W_r, W_n, h0j, [noisej])  # Forward pass

        # DYNAMICS
        hj_mu = forward_inj(W_r, W_n, h0j, noise_vec_j, ret_intermediate=True)  
        hj_mu = np.array(hj_mu)
        out = (np.einsum("ij, bj -> bi", W_o, hj_mu)).argmax(axis=-1)
        pca_hj = PCA(n_components=2)
        pca_hj.fit(hj_mu)
        pca_comps.append(pca_hj.components_)
        pca_points.append(hj_mu)
        out_t.append(out)

        # PERTURBATION
        exdh2_j, exdh1_j = expected_deltah(W_r, W_n, h0j, trials=trials, t0=t0, tf=tdh2, sigma=sigmaj, h_eps=h_eps, silent=True)
        vec_dh2_l.append(exdh2_j)
        vec_dh1_l.append(exdh1_j)

        # JACOBIANS
        Jt_l = []
        for t in range(len(hj_mu)-1):
            Jt_ = jacobian_inj(W_r, hj_mu[t+1], W_n, noise_vec_j[t], ret_D=False)
            Jt_l.append(Jt_)
        Jt_l = np.array(Jt_l)

        # HOPF EIGENVALUES
        cms_max, cms_m, cms_s, us_max, us_m, us_s = Hopf_eigs_stat(Jt_l, do_mobius=True, t0=t0, eps_real=0.1, eps_img=0.05)
        compl_stats_max.append(cms_max)
        compl_stats_mean.append(cms_m)
        compl_stats_std.append(cms_s)
        unst_stats_max.append(us_max)
        unst_stats_mean.append(us_m)
        unst_stats_std.append(us_s)

        try:
            a = np.where(np.abs(np.diff(out>0)))[0]
            tr = a[1:][np.argmax(np.diff(a[1:]))]-3
            avg_res = np.diff(np.where(np.diff(out)!=0)[0:]).mean()
            avg_res_time.append(avg_res)
            avg_firing_rate.append((1/np.diff(np.where(np.diff(out)!=0)[0:]).flatten()).mean())
        except:
            tr = T//2
            avg_res_time.append(T)
            avg_firing_rate.append(0)
            
    vec_dh2_l = np.array(vec_dh2_l)
    vec_dh1_l = np.array(vec_dh1_l)
    compl_stats_max = np.array(compl_stats_max)
    compl_stats_mean = np.array(compl_stats_mean)
    compl_stats_std = np.array(compl_stats_std)
    avg_res_time = np.array(avg_res_time)
    avg_firing_rate = np.array(avg_firing_rate)
    pca_comps = np.array(pca_comps)
    pca_points = np.array(pca_points)
    unst_stats_max = np.array(unst_stats_max)
    unst_stats_mean = np.array(unst_stats_mean)
    unst_stats_std = np.array(unst_stats_std)
    out_t = np.array(out_t)

    if bifep is None:
        bifep = np.where(np.abs(np.diff(compl_stats_mean))>0.02)[0][0]*ep_rate

    font = {'size'   : 16}
    matplotlib.rc('font', **font)

    plt.figure(figsize=(20, 6))
    plt.subplot(1, 3, 1)
    plt.plot(np.arange(0, len(compl_stats_mean))*ep_rate, avg_firing_rate, color="darkorange")


    plt.axhline(np.mean(avg_firing_rate[-6:]), color='red', lw=1.5, ls='--', label=f"converged TR")
    plt.title(f"Average transition rate")
    plt.xlabel("Epochs")
    plt.ylabel("Transition rate")

    plt.axvline(bifep, color='darkviolet', lw=1.5, ls='--', label="bifurcation")
    plt.legend()
    plt.subplot(1, 3, 2)

    plt.plot(np.arange(0, len(compl_stats_mean))*ep_rate, compl_stats_mean, label="complex", color="darkorange")
    plt.plot(np.arange(0, len(unst_stats_mean))*ep_rate, unst_stats_mean, label="unstable", color="darkred")
    plt.title("Complex and unstable eigenvalues")
    plt.xlabel("Epochs")
    plt.ylabel("Mean eignvalues count")
    plt.legend()

    plt.axvline(bifep, color='darkviolet', lw=1.5, ls='--')

    plt.subplot(1, 3, 3)
    plt.title("Expected $dh^{(2)}$")
    plt.imshow(np.sqrt(np.abs(vec_dh2_l)).T, aspect='auto', cmap='Reds', interpolation='nearest',extent=[0, vec_dh2_l.shape[1], vec_dh2_l.shape[0]*ep_rate, 0])

    cmap = plt.cm.inferno
    cmap.set_bad(color='white')  # Color for masked values (below threshold)
    masked_data = np.ma.masked_where(np.sqrt(np.abs(vec_dh2_l)).T <= threshold, vec_dh2_l.T)
    plt.imshow(masked_data, aspect='auto', cmap=cmap, interpolation='nearest',
            extent=[0, vec_dh2_l.shape[0]*ep_rate, vec_dh2_l.shape[1], 0])

    plt.colorbar(label='$\mathbb{E}[dh^{(2)}]$')

    plt.yticks([])
    plt.xlabel("Epochs")
    plt.ylabel("Hidden units")
    plt.axvline(bifep, color='darkviolet', lw=1.5, ls='--')
    plt.show()
    

def ablation(weights_path, group=0, verbose=False):
    if group == 0:
        ablated_neurons = [83, 59, 6]
    elif group == 1:
        ablated_neurons = [114, 72, 28]

    W_r, W_n, W_o = load_weights(weights_path)
    h_nom, o_track, noise_vec_nom, tt, tk, tc = locate_areas(W_r, W_n, W_o, verbose=verbose)
    
    # Parameters
    T = 1000
    muvals = np.arange(0, 2.1, 0.1).tolist()
    trajs = 5
    trajlen = 100
    sigmaj = 1
    gamm_noise = 0.5
    
    # Define intervention masks
    # Kick neurons
    c_kick = np.zeros(W_r.shape[0])
    c_kick[ablated_neurons] = 1
    cn_kick = np.zeros(W_r.shape[0])
    
    # Noise projection
    c_noise = np.zeros(W_r.shape[0])
    cn_noise = np.zeros(W_r.shape[0])
    mask = np.zeros(W_r.shape[0]).astype(bool)
    dpop_size = 140 # size of population 1 + population 2
    g1 = W_r[[83,59,6],:].mean(axis=0)
    g2 = W_r[[28,72,114],:].mean(axis=0)
    sortmap = np.argsort(g1*g2)
    g1 = g1[sortmap][:dpop_size]
    g2 = g2[sortmap][:dpop_size]

    if group==0:
        sortmap2 = np.argsort(g1)
        nz_vals = (g1>0).sum()
    else:
        sortmap2 = np.argsort(-g2)
        nz_vals = (g2>0).sum()
    final_map = sortmap[sortmap2[sortmap2<dpop_size]]
    top_k = 83
    top_k = min(top_k, nz_vals)
    mask[final_map[:top_k]] = True
    cn_noise[np.where(mask)] = 1
    cn_noise[[28, 72, 114, 59, 6, 83]] = 0
    
    # Control
    c_cnt = np.zeros(W_r.shape[0])
    cn_cnt = np.zeros(W_r.shape[0])
    top_k = 35
    mask = np.any(np.stack([
        kprojected(W_r, 83, top_k=top_k, reversed=True),
        kprojected(W_r, 59, top_k=top_k, reversed=True),
        kprojected(W_r, 6, top_k=top_k, reversed=True),
        kprojected(W_r, 28, top_k=top_k, reversed=True),
        kprojected(W_r, 72, top_k=top_k, reversed=True),
        kprojected(W_r, 114, top_k=top_k, reversed=True)
    ]), axis=0).flatten()
    mask = ~mask
    cn_cnt[np.where(mask)] = 1
    cn_cnt[[28, 72, 114, 59, 6, 83]] = 0
    
    # Run simulations
    mu_data_kick = run_injected(W_r, W_n, W_o, tc, h_nom, noise_vec_nom, trajs, trajlen, muvals, c_kick, cn_kick, T, sigmaj, gamm_noise)
    mu_data_noise = run_injected(W_r, W_n, W_o, tc, h_nom, noise_vec_nom, trajs, trajlen, muvals, c_noise, cn_noise, T, sigmaj, gamm_noise)
    mu_data_cnt = run_injected(W_r, W_n, W_o, tc, h_nom, noise_vec_nom, trajs, trajlen, muvals, c_cnt, cn_cnt, T, sigmaj, gamm_noise)
    
    # Compute PCA
    pca_hj = PCA(n_components=2)
    pca_hj.fit(h_nom)
    for mu in muvals:
        for mu_data in [mu_data_kick, mu_data_noise, mu_data_cnt]:
            mu_data[mu]["h_istj_pca"] = pca_hj.transform(mu_data[mu]["h_istj_mu"])
            mu_data[mu]["traj_l_pca"] = pca_hj.transform(mu_data[mu]["traj_l"].reshape(-1, W_r.shape[0])).reshape(trajs, trajlen + 1, 2)
    
    # Compute critical eigenvalues
    critical_complex_eigs_mean_kick, critical_complex_eigs_std_kick = compute_critical_eigs(mu_data_kick, muvals)
    critical_complex_eigs_mean_noise, critical_complex_eigs_std_noise = compute_critical_eigs(mu_data_noise, muvals)
    critical_complex_eigs_mean_cnt, critical_complex_eigs_std_cnt = compute_critical_eigs(mu_data_cnt, muvals)
    
    # Plotting setup
    plt.rcParams['font.size'] = 16
    colors = {0: 'darkgreen', 1: "royalblue", 2: 'darkred'}
    cmap = mcolors.ListedColormap([colors[0], colors[2]])
    max_trajs = 3
    mu0 = 0
    muf = muvals[-2]  # 1.9
    a_inhi = 0.7
    a_exci = 0.7
    a_nomi = 0.7
    a_hopf = 0.3
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # Kick neurons
    axs[0, 0].set_title("Kick neurons")
    for k in range(min(max_trajs, len(mu_data_kick[mu0]["traj_l_pca"]))):
        axs[0, 0].plot(mu_data_kick[1]["traj_l_pca"][k][:60, 0], mu_data_kick[1]["traj_l_pca"][k][:60, 1], c="gray", alpha=a_nomi, label="nominal" if k == 0 else "")
        axs[0, 0].plot(mu_data_kick[mu0]["traj_l_pca"][k][:60, 0], mu_data_kick[mu0]["traj_l_pca"][k][:60, 1], color="darkblue", alpha=a_inhi, label="inhibition" if k == 0 else "")
        axs[0, 0].plot(mu_data_kick[muf]["traj_l_pca"][k][:60, 0], mu_data_kick[muf]["traj_l_pca"][k][:60, 1], color="darkred", alpha=a_exci, label="excitation" if k == 0 else "")
    axs[0, 0].scatter(mu_data_kick[mu0]["traj_l_pca"][0][0, 0], mu_data_kick[mu0]["traj_l_pca"][0][0, 1], c='black', alpha=1, marker="*", s=100)
    axs[0, 0].set_xlabel("PC1")
    axs[0, 0].set_ylabel("PC2")
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    
    axs[1, 0].scatter(muvals[:-2], list(critical_complex_eigs_mean_kick.values())[:-2], color='darkgray', alpha=0.5)
    axs[1, 0].plot(muvals[:-2], list(critical_complex_eigs_mean_kick.values())[:-2], color='darkgray', alpha=0.2, linestyle='--')
    axs[1, 0].errorbar(muvals[:-2], list(critical_complex_eigs_mean_kick.values())[:-2], yerr=list(critical_complex_eigs_std_kick.values())[:-2], fmt='o', color='darkgray', alpha=a_hopf)
    axs[1, 0].errorbar(muf, critical_complex_eigs_mean_kick[muf], yerr=critical_complex_eigs_std_kick[muf], fmt='o', color='darkred', alpha=0.8)
    axs[1, 0].errorbar(mu0, critical_complex_eigs_mean_kick[mu0], yerr=critical_complex_eigs_std_kick[mu0], fmt='o', color='darkblue', alpha=a_inhi)
    axs[1, 0].errorbar(1, critical_complex_eigs_mean_kick[1], yerr=critical_complex_eigs_std_kick[1], fmt='o', color='black', alpha=0.45)
    axs[1, 0].set_xticks([0, 0.5, 1, 1.5, 2])
    axs[1, 0].tick_params("x", rotation=45)
    axs[1, 0].set_xlabel("$\mu$")
    axs[1, 0].set_ylabel("# Critical pairs")
    
    # Noise projection
    axs[0, 1].set_title("Noise projection")
    for k in range(min(max_trajs, len(mu_data_noise[mu0]["traj_l_pca"]))):
        axs[0, 1].plot(mu_data_kick[1]["traj_l_pca"][k][:60, 0], mu_data_kick[1]["traj_l_pca"][k][:60, 1], c="gray", alpha=a_nomi, label="nominal" if k == 0 else "")
        axs[0, 1].plot(mu_data_noise[mu0]["traj_l_pca"][k][:60, 0], mu_data_noise[mu0]["traj_l_pca"][k][:60, 1], color="darkblue", alpha=a_inhi, label="inhibition" if k == 0 else "")
        axs[0, 1].plot(mu_data_noise[muf]["traj_l_pca"][k][:60, 0], mu_data_noise[muf]["traj_l_pca"][k][:60, 1], color="darkred", alpha=a_exci, label="excitation" if k == 0 else "")
    axs[0, 1].scatter(mu_data_noise[mu0]["traj_l_pca"][0][0, 0], mu_data_noise[mu0]["traj_l_pca"][0][0, 1], c='black', alpha=1, marker="*", s=100)
    axs[0, 1].set_xlabel("PC1")
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    
    axs[1, 1].scatter(muvals[:-2], list(critical_complex_eigs_mean_noise.values())[:-2], color='darkgray', alpha=0.5)
    axs[1, 1].plot(muvals[:-2], list(critical_complex_eigs_mean_noise.values())[:-2], color='darkgray', alpha=0.2, linestyle='--')
    axs[1, 1].errorbar(muvals[:-2], list(critical_complex_eigs_mean_noise.values())[:-2], yerr=list(critical_complex_eigs_std_noise.values())[:-2], fmt='o', color='darkgray', alpha=a_hopf)
    axs[1, 1].errorbar(muf, critical_complex_eigs_mean_noise[muf], yerr=critical_complex_eigs_std_noise[muf], fmt='o', color='darkred', alpha=0.8)
    axs[1, 1].errorbar(mu0, critical_complex_eigs_mean_noise[mu0], yerr=critical_complex_eigs_std_noise[mu0], fmt='o', color='darkblue', alpha=a_inhi)
    axs[1, 1].errorbar(1, critical_complex_eigs_mean_noise[1], yerr=critical_complex_eigs_std_noise[1], fmt='o', color='black', alpha=0.45)
    axs[1, 1].set_xticks([0, 0.5, 1, 1.5, 2])
    axs[1, 1].tick_params("x", rotation=45)
    axs[1, 1].set_xlabel("$\mu$")
    
    # Control
    axs[0, 2].set_title("Control")
    for k in range(min(max_trajs, len(mu_data_cnt[mu0]["traj_l_pca"]))):
        axs[0, 2].plot(mu_data_kick[1]["traj_l_pca"][k][:60, 0], mu_data_kick[1]["traj_l_pca"][k][:60, 1], c="gray", alpha=a_nomi, label="nominal" if k == 0 else "")
        axs[0, 2].plot(mu_data_cnt[mu0]["traj_l_pca"][k][:60, 0], mu_data_cnt[mu0]["traj_l_pca"][k][:60, 1], color="darkblue", alpha=a_inhi, label="inhibition" if k == 0 else "")
        axs[0, 2].plot(mu_data_cnt[muf]["traj_l_pca"][k][:60, 0], mu_data_cnt[muf]["traj_l_pca"][k][:60, 1], color="darkred", alpha=a_exci, label="excitation" if k == 0 else "")
    axs[0, 2].scatter(mu_data_cnt[mu0]["traj_l_pca"][0][0, 0], mu_data_cnt[mu0]["traj_l_pca"][0][0, 1], c='black', alpha=1, marker="*", s=100)
    axs[0, 2].set_xlabel("PC1")
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    axs[0, 2].legend(loc="upper left", fontsize=12, bbox_to_anchor=(1.05, 0.0), borderaxespad=0.)
    
    axs[1, 2].scatter(muvals[:-2], list(critical_complex_eigs_mean_cnt.values())[:-2], color='darkgray', alpha=0.5)
    axs[1, 2].plot(muvals[:-2], list(critical_complex_eigs_mean_cnt.values())[:-2], color='darkgray', alpha=0.2, linestyle='--')
    axs[1, 2].errorbar(muvals[:-2], list(critical_complex_eigs_mean_cnt.values())[:-2], yerr=list(critical_complex_eigs_std_cnt.values())[:-2], fmt='o', color='darkgray', alpha=a_hopf)
    axs[1, 2].errorbar(muf, critical_complex_eigs_mean_cnt[muf], yerr=critical_complex_eigs_std_cnt[muf], fmt='o', color='darkred', alpha=0.8)
    axs[1, 2].errorbar(mu0, critical_complex_eigs_mean_cnt[mu0], yerr=critical_complex_eigs_std_cnt[mu0], fmt='o', color='darkblue', alpha=a_inhi)
    axs[1, 2].errorbar(1, critical_complex_eigs_mean_cnt[1], yerr=critical_complex_eigs_std_cnt[1], fmt='o', color='black', alpha=0.45)
    axs[1, 2].set_xticks([0, 0.5, 1, 1.5, 2])
    axs[1, 2].tick_params("x", rotation=45)
    axs[1, 2].set_xlabel("$\mu$")
    
    plt.tight_layout()
    plt.show()
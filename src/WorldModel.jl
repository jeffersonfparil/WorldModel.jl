# ==============================================================================
#           DRAFT CODE: TRAINING A WORLD MODEL IN JULIA
#
# This script outlines the core components and training loops for a
# simplified World Model architecture, inspired by the 2018 paper by
# Ha & Schmidhuber.
#
# Components:
# 1. Convolutional VAE: Properly handles image data.
# 2. Probabilistic RNN: Uses a Mixture Density Network concept to handle uncertainty.
# 3. Controller Training Loop: Implements a conceptual RL "imagination" loop.
# 4. Replay Buffer: Manages experience collected from an environment.
# 5. GPU Support: Automatically utilizes a GPU if available.
#
# Notes:
# 1. Algorithmic and Theoretical Simplifications
#     - Imagined Reward Function is Arbitrary: This is a critical flaw. The controller's loss is based on an imagined_reward that is just the mean(z_next). This is a placeholder that doesn't correspond to any real-world objective. The agent is learning to maximize a meaningless metric.
#         Improvement: In a real task, the reward function must be meaningful. You would either need to: a. Define the reward based on the task (e.g., in a racing game, reward is speed). b. Train another small neural network to predict the reward from the latent state z, so the agent can optimize for predicted rewards in its "dream".
#     - Improper Sampling from the Memory Model (MDN-RNN):
#         In the imagination loop, the code "samples" the next state by picking the mean (μ) of the most likely Gaussian component. This is a deterministic shortcut that ignores the model's learned uncertainty. The model might learn that a state can have three very different but equally likely outcomes, but the agent will only ever "see" one.
#         Improvement: You must perform true sampling from the Gaussian Mixture Model. This involves first sampling a Gaussian component based on the mixture weights (πs), and then sampling a z vector from that chosen Gaussian's μs and σs. This makes the "dream" stochastic and allows the agent to explore a wider range of possible futures.
#     - Controller Lacks Exploration:
#         The controller's output is deterministic. When acting, it will always pick the same action for the same state. This can cause it to get stuck in a suboptimal policy because it never tries anything new.
#         Improvement: Introduce exploration, typically by adding noise to the action output. For example, you could add small, random noise from a Gaussian distribution to the action selected by the controller.
# 2. Implementation and Training Loop Deficiencies
#     - Episode Termination (done flag) is Ignored:
#         The RNN is trained on sequences of (z_t, a_t) -> z_{t+1}. However, if an episode ends at t, then z_{t+1} is the start of a new, unrelated episode. The RNN is being incorrectly taught to predict a connection where none exists.
#         Improvement: The training loop must respect episode boundaries. When sampling sequences from the replay buffer, you should either ensure they don't cross done flags or use a mask to signal to the loss function that the prediction for the step after done should be ignored.
#     - Rigid, Phased Training:
#         The script follows a strict sequence: 1. Collect all data -> 2. Train VAE/RNN -> 3. Train Controller. This is inefficient and not how real agents learn.
#         Improvement: The training should be interleaved and continuous. The main loop should look more like this:
#             Act in the environment for N steps using the current controller, adding experiences to the buffer.
#             Perform one or more training steps for the VAE, RNN, and Controller on batches sampled from the buffer.
#             Repeat. This allows the agent to learn from the data it gathers with its own improving policy.
#     - Potential for Vanishing/Exploding Gradients:
#         The imagination loop runs for imagination_horizon steps. Backpropagating the reward signal through a 15-step (or longer) chain of RNN predictions is prone to gradient issues.
#         Improvement: Use techniques like gradient clipping to prevent exploding gradients. For very long horizons, more advanced reinforcement learning algorithms like PPO or methods that use a learned value function (like DreamerV2) are more stable.
# 3. System Design and Practicality
#     - Hyperparameter Sensitivity:
#         The script contains many "magic numbers" (learning rates, latent_dim, num_gaussians, the weight of the VAE's KL-divergence term, etc.). These parameters are highly interdependent and sensitive. The current values are guesses and would require extensive, systematic tuning to get the model to learn effectively.
#         Improvement: A real project would involve a rigorous hyperparameter search using tools like HyperOpt or Optuna.
#     - Simulated Environment is a Toy:
#         The environment that generates a random image with a white square is too simple. It lacks any concept of physics, momentum, or consistent cause-and-effect. The models would not be able to learn any meaningful world dynamics from it.
#         Improvement: Use a standard benchmark environment, like one from Gymnasium.jl in Julia, or a more complex physics simulator to test the architecture on a real problem.
# ==============================================================================

using Flux
using Flux: @functor, mse, throttle, params, glorot_normal
using Zygote
using Distributions # For Mixture Density Network
using Statistics
using Random
using Printf
using CircularArrayBuffers # For the replay buffer

println("World Model in Julia")
println("="^50)

# --- GPU Device Configuration ---
const device = gpu_device()
println("Training device: ", device)

# --- Ensure reproducibility
Random.seed!(1234)

# ==============================================================================
# 0. REPLAY BUFFER AND SIMULATED ENVIRONMENT
# ==============================================================================
# In a real scenario, this would be a sophisticated environment (e.g., a game).
# We simulate it to provide data for the replay buffer.

struct SimulatedEnv
    obs_size::Tuple
    action_dim::Int
end

function get_obs(env::SimulatedEnv)
    # Simulate a 64x64x3 image with some "structure"
    obs = rand(Float32, env.obs_size...)
    c1, c2 = rand(1:env.obs_size[1]-10), rand(1:env.obs_size[2]-10)
    obs[c1:c1+5, c2:c2+5, :] .= 1.0f0
    return obs
end

function step!(env::SimulatedEnv, action)
    # Dummy step function
    next_obs = get_obs(env)
    reward = randn()
    done = rand() > 0.98
    return next_obs, reward, done
end

# Replay buffer to store experiences
const buffer_size = 10000
const obs_shape = (64, 64, 3)
const action_dim = 3
const replay_buffer = CircularArrayBuffer{NamedTuple{(:obs, :action, :reward, :next_obs, :done), NTuple{5, Any}}}(buffer_size)

# ==============================================================================
# 1. VISION MODEL (V) - CONVOLUTIONAL VAE
# ==============================================================================
const latent_dim = 32

struct ConvVAE
    encoder
    μ
    logσ
    decoder
end
@functor ConvVAE

# Reparameterization trick
reparameterize(μ, logσ) = μ .+ exp.(logσ) .* randn(Float32, size(μ)) |> device

function ConvVAE(latent_dim::Int)
    # Input: 64x64x3
    encoder_conv = Chain(
        Conv((4, 4), 3 => 32, relu; stride=2, pad=1), # -> 32x32x32
        Conv((4, 4), 32 => 64, relu; stride=2, pad=1), # -> 16x16x64
        Conv((4, 4), 64 => 128, relu; stride=2, pad=1), # -> 8x8x128
        Conv((4, 4), 128 => 256, relu; stride=2, pad=1), # -> 4x4x256
        Flux.flatten
    )

    # Calculate flattened size
    dummy_input = randn(Float32, 64, 64, 3, 1)
    flattened_size = size(encoder_conv(dummy_input), 1)

    encoder = Chain(encoder_conv, Dense(flattened_size, 256, relu))

    μ = Dense(256, latent_dim)
    logσ = Dense(256, latent_dim)

    decoder_dense = Dense(latent_dim, flattened_size, relu)
    decoder_conv = Chain(
        x -> reshape(x, 4, 4, 256, :),
        ConvTranspose((4, 4), 256 => 128, relu; stride=2, pad=1), # -> 8x8x128
        ConvTranspose((4, 4), 128 => 64, relu; stride=2, pad=1),  # -> 16x16x64
        ConvTranspose((4, 4), 64 => 32, relu; stride=2, pad=1),   # -> 32x32x32
        ConvTranspose((4, 4), 32 => 3, sigmoid; stride=2, pad=1)  # -> 64x64x3, sigmoid for [0,1]
    )
    decoder = Chain(decoder_dense, decoder_conv)

    return ConvVAE(encoder, μ, logσ, decoder)
end

function (m::ConvVAE)(x)
    h = m.encoder(x)
    μ_val, logσ_val = m.μ(h), m.logσ(h)
    z = reparameterize(μ_val, logσ_val)
    x_recon = m.decoder(z)
    return x_recon, μ_val, logσ_val
end

function vae_loss(m::ConvVAE, x)
    x_recon, μ, logσ = m(x)
    recon_loss = mse(x_recon, x)
    kl_div = -0.5f0 * mean(1.0f0 .+ 2.0f0 .* logσ .- μ.^2 .- exp.(2.0f0 .* logσ))
    return recon_loss + kl_div * 1.0f-4
end

# ==============================================================================
# 2. MEMORY MODEL (M) - PROBABILISTIC RNN (MDN-RNN)
# ==============================================================================
const hidden_rnn_dim = 256
const num_gaussians = 5 # Number of components in the mixture model

# MDN-RNN predicts parameters for a Gaussian Mixture Model for the next latent state
struct MDN_RNN
    rnn
    μ_head
    σ_head
    π_head # Mixture weights
end
@functor MDN_RNN

function MDN_RNN(latent_dim::Int, action_dim::Int, hidden_dim::Int, n_gaussians::Int)
    input_size = latent_dim + action_dim
    rnn = LSTM(input_size, hidden_dim)
    # Each head outputs n_gaussians * latent_dim parameters
    μ_head = Dense(hidden_dim, n_gaussians * latent_dim)
    σ_head = Dense(hidden_dim, n_gaussians * latent_dim)
    π_head = Chain(Dense(hidden_dim, n_gaussians), softmax)
    return MDN_RNN(rnn, μ_head, σ_head, π_head)
end

function (m::MDN_RNN)(z, a)
    rnn_input = vcat(z, a)
    # RNN needs input shape [features, sequence_len=1, batch_size]
    rnn_input_seq = reshape(rnn_input, :, 1, size(z, 2))
    h = m.rnn(rnn_input_seq)
    
    # Reshape to [latent_dim, n_gaussians, batch_size] for easier processing
    μs = reshape(m.μ_head(h), latent_dim, :, size(z, 2))
    σs = exp.(reshape(m.σ_head(h), latent_dim, :, size(z, 2))) # Ensure std dev is positive
    πs = m.π_head(h) # Shape [n_gaussians, batch_size]
    
    return μs, σs, πs
end

# Loss for the MDN-RNN is the negative log-likelihood of the actual next state
function mdn_rnn_loss(m::MDN_RNN, z_t, a_t, z_t_next)
    μs, σs, πs = m(z_t, a_t)
    
    # Create a batch of mixture models
    # Zygote struggles with distributions, so we calculate log-likelihood manually
    # This is a simplified version; a real one would be more careful with stability
    total_log_prob = 0.0f0
    for i in 1:size(z_t, 2) # Iterate over batch
        log_probs_gaussians = -sum(0.5f0 .* ((z_t_next[:, i] .- μs[:,:,i])./σs[:,:,i]).^2 .+ log.(σs[:,:,i]) .+ 0.5f0*log(2.0f0*π), dims=1)
        # Log-sum-exp for stability
        max_log_prob = maximum(log_probs_gaussians)
        log_prob_mix = max_log_prob + log(sum(πs[:, i] .* exp.(log_probs_gaussians .- max_log_prob)))
        total_log_prob += log_prob_mix
    end

    return -total_log_prob / size(z_t, 2) # Return negative log-likelihood
end

# ==============================================================================
# 3. CONTROLLER MODEL (C) - THE AGENT
# ==============================================================================
struct Controller
    linear
end
@functor Controller

function Controller(latent_dim::Int, hidden_rnn_dim::Int, action_dim::Int)
    input_size = latent_dim + hidden_rnn_dim
    linear = Dense(input_size, action_dim, tanh) # tanh to bound actions in [-1, 1]
    return Controller(linear)
end

# The controller acts based on the current latent state `z` and the RNN's memory `h`
function (c::Controller)(z, h)
    return c.linear(vcat(z, h))
end

# ==============================================================================
# 4. TRAINING SCRIPT
# ==============================================================================
function run_training()
    println("\n--- Starting Improved World Model Training ---")

    # --- Hyperparameters ---
    epochs = 1
    controller_train_iterations = 100
    imagination_horizon = 15
    batch_size = 32
    lr = 1e-4

    # --- Initialize Models and move to GPU ---
    vae_model = ConvVAE(latent_dim) |> device
    rnn_model = MDN_RNN(latent_dim, action_dim, hidden_rnn_dim, num_gaussians) |> device
    controller_model = Controller(latent_dim, hidden_rnn_dim, action_dim) |> device

    # --- Optimizers ---
    opt_vae = ADAM(lr)
    opt_rnn = ADAM(lr)
    opt_controller = ADAM(lr * 0.1)

    println("Models and optimizers initialized on $device.")

    # --- PHASE 1: Data Collection (Random Policy) ---
    println("\n--- Populating Replay Buffer with Random Actions ---")
    sim_env = SimulatedEnv(obs_shape, action_dim)
    current_obs = get_obs(sim_env)
    for _ in 1:2000
        action = randn(Float32, action_dim)
        next_obs, reward, done = step!(sim_env, action)
        push!(replay_buffer, (obs=current_obs, action=action, reward=reward, next_obs=next_obs, done=done))
        current_obs = done ? get_obs(sim_env) : next_obs
    end
    println("Replay buffer populated with $(length(replay_buffer)) experiences.")

    # --- PHASE 2: Train VAE and RNN ---
    println("\n--- Training VAE and RNN ---")
    for epoch in 1:epochs
        # --- Train VAE ---
        batch = rand(replay_buffer, batch_size)
        obs_batch = cat([b.obs for b in batch]...; dims=4) |> device
        
        vae_loss_val, grads_vae = withgradient(m -> vae_loss(m, obs_batch), vae_model)
        Flux.update!(opt_vae, params(vae_model), grads_vae[1])

        # --- Train RNN ---
        next_obs_batch = cat([b.next_obs for b in batch]...; dims=4) |> device
        action_batch = hcat([b.action for b in batch]...) |> device
        
        # Get latent states from VAE (we don't train VAE here, so stop gradient)
        Zygote.ignore() do
            global z_t = vae_model.μ(vae_model.encoder(obs_batch))
            global z_t_next = vae_model.μ(vae_model.encoder(next_obs_batch))
        end

        rnn_loss_val, grads_rnn = withgradient(m -> mdn_rnn_loss(m, z_t, action_batch, z_t_next), rnn_model)
        Flux.update!(opt_rnn, params(rnn_model), grads_rnn[1])
        
        if epoch % 100 == 0
             @printf("Epoch: %d, VAE Loss: %.4f, RNN Loss: %.4f\n", epoch, vae_loss_val, rnn_loss_val)
        end
    end

    # --- PHASE 3: Train Controller via "Imagination" ---
    println("\n--- Training Controller in Imagination ---")
    for iter in 1:controller_train_iterations
        # 1. Get a starting state from real data
        batch = rand(replay_buffer, batch_size)
        obs_batch = cat([b.obs for b in batch]...; dims=4) |> device
        
        # 2. Get initial latent state `z` and RNN hidden state `h`
        z0 = Zygote.ignore(() -> vae_model.μ(vae_model.encoder(obs_batch)))
        h_rnn = Zygote.ignore(() -> rnn_model.rnn.state[1]) # Get initial LSTM hidden state
        if typeof(h_rnn) <: Tuple
             h_rnn = h_rnn[1] # For LSTM, state is a tuple (h, c)
        end
        h_rnn = repeat(h_rnn, 1, batch_size) # Match batch size

        # 3. Define the loss function for the controller
        # We want to maximize the imagined future reward
        function controller_loss(c_model)
            imagined_reward = 0.0f0
            z_current = z0
            h_current = h_rnn

            # Imagine a trajectory for `horizon` steps
            for _ in 1:imagination_horizon
                # Get action from controller
                action = c_model(z_current, h_current)
                
                # Predict next state with RNN
                μs, σs, πs = rnn_model(z_current, action)
                
                # Sample from the mixture to get the next state
                # (simplified: just take the mean of the most likely component)
                best_mix_idx = Flux.onecold(πs)
                z_next = [μs[:, best_mix_idx[i], i] for i in 1:batch_size]
                z_next = hcat(z_next...)

                # Update RNN hidden state
                rnn_input = vcat(z_next, action)
                rnn_input_seq = reshape(rnn_input, :, 1, size(z_next, 2))
                rnn_model.rnn(rnn_input_seq) # Update internal state
                h_next = rnn_model.rnn.state[1]
                if typeof(h_next) <: Tuple
                    h_next = h_next[1]
                end

                # Simple reward function: encourage moving "forward" in latent space
                imagined_reward += mean(z_next) 
                
                z_current, h_current = z_next, h_next
            end
            
            return -imagined_reward # We minimize the negative reward
        end

        # 4. Calculate gradients and update the controller
        ctrl_loss_val, grads_ctrl = withgradient(m -> controller_loss(m), controller_model)
        Flux.update!(opt_controller, params(controller_model), grads_ctrl[1])
        
         if iter % 10 == 0
             @printf("Controller Iteration: %d, Imagined Reward: %.4f\n", iter, -ctrl_loss_val)
        end
    end
    
    println("\n--- Training Simulation Finished ---")
end


# --- Run the main training script ---
run_training()

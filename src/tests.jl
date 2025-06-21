@testset "World Model Unit Tests" begin

    @testset "SimulatedEnv Tests" begin
        env = SimulatedEnv((64, 64, 3), 3)
        @test env.obs_size == (64, 64, 3)
        @test env.action_dim == 3

        obs = get_obs(env)
        @test size(obs) == (64, 64, 3)
        @test eltype(obs) == Float32
        @test all(0.0f0 .<= obs .<= 1.0f0) # Check if values are normalized

        action = randn(Float32, env.action_dim)
        next_obs, reward, done = step!(env, action)
        @test size(next_obs) == (64, 64, 3)
        @test typeof(reward) == Float64 # randn() returns Float64
        @test typeof(done) == Bool
    end

    @testset "Replay Buffer Tests" begin
        buffer = CircularArrayBuffer{NamedTuple{(:obs, :action, :reward, :next_obs, :done), NTuple{5, Any}}}(5)
        @test isempty(buffer)
        @test length(buffer) == 0

        # Add some dummy data
        for i in 1:3
            push!(buffer, (obs=zeros(Float32, 1,1,1), action=[0.0f0], reward=0.0f0, next_obs=zeros(Float32, 1,1,1), done=false))
        end
        @test length(buffer) == 3
        @test !isempty(buffer)

        # Test overflow
        for i in 1:3
            push!(buffer, (obs=zeros(Float32, 1,1,1), action=[0.0f0], reward=0.0f0, next_obs=zeros(Float32, 1,1,1), done=false))
        end
        @test length(buffer) == 5 # Should not exceed buffer_size

        # Test sampling
        sample_batch = rand(buffer, 2)
        @test length(sample_batch) == 2
        @test isa(sample_batch, Vector{<:NamedTuple})
    end

    @testset "ConvVAE Tests" begin
        vae = ConvVAE(latent_dim) |> device
        @test typeof(vae) == ConvVAE
        @test vae.encoder isa Chain
        @test vae.μ isa Dense
        @test vae.logσ isa Dense
        @test vae.decoder isa Chain

        input_image = rand(Float32, obs_shape..., 1) |> device # Batch size of 1
        x_recon, μ, logσ = vae(input_image)

        @test size(x_recon) == size(input_image)
        @test size(μ) == (latent_dim, 1)
        @test size(logσ) == (latent_dim, 1)
        @test eltype(x_recon) == Float32
        @test eltype(μ) == Float32
        @test eltype(logσ) == Float32

        # Test vae_loss
        loss_val = vae_loss(vae, input_image)
        @test typeof(loss_val) == Float32
        @test loss_val > 0.0f0

        # Test gradient computation
        grads = Zygote.gradient(m -> vae_loss(m, input_image), vae)
        @test grads[1] !== nothing # Check that gradients are computed
    end

    @testset "MDN_RNN Tests" begin
        rnn = MDN_RNN(latent_dim, action_dim, hidden_rnn_dim, num_gaussians) |> device
        @test typeof(rnn) == MDN_RNN
        @test rnn.rnn isa LSTM
        @test rnn.μ_head isa Dense
        @test rnn.σ_head isa Dense
        @test rnn.π_head isa Chain

        batch_size_test = 2
        z_input = rand(Float32, latent_dim, batch_size_test) |> device
        a_input = rand(Float32, action_dim, batch_size_test) |> device

        μs, σs, πs = rnn(z_input, a_input)

        @test size(μs) == (latent_dim, num_gaussians, batch_size_test)
        @test size(σs) == (latent_dim, num_gaussians, batch_size_test)
        @test size(πs) == (num_gaussians, batch_size_test)
        @test all(σs .> 0.0f0) # Standard deviations must be positive
        @test all(sum(πs; dims=1) .≈ 1.0f0) # Mixture weights must sum to 1

        # Test mdn_rnn_loss
        z_next_input = rand(Float32, latent_dim, batch_size_test) |> device
        loss_val = mdn_rnn_loss(rnn, z_input, a_input, z_next_input)
        @test typeof(loss_val) == Float32
        @test loss_val > 0.0f0 # NLL should be positive

        # Test gradient computation
        grads = Zygote.gradient(m -> mdn_rnn_loss(m, z_input, a_input, z_next_input), rnn)
        @test grads[1] !== nothing
    end

    @testset "Controller Tests" begin
        controller = Controller(latent_dim, hidden_rnn_dim, action_dim) |> device
        @test typeof(controller) == Controller
        @test controller.linear isa Dense

        batch_size_test = 2
        z_input = rand(Float32, latent_dim, batch_size_test) |> device
        # For LSTM, hidden state is typically a matrix [hidden_dim, batch_size]
        h_input = rand(Float32, hidden_rnn_dim, batch_size_test) |> device

        action_output = controller(z_input, h_input)

        @test size(action_output) == (action_dim, batch_size_test)
        @test eltype(action_output) == Float32
        @test all(-1.0f0 .<= action_output .<= 1.0f0) # Due to tanh activation

        # Test gradient computation for a dummy loss
        dummy_loss(c) = mse(c(z_input, h_input), zeros(Float32, action_dim, batch_size_test) |> device)
        grads = Zygote.gradient(dummy_loss, controller)
        @test grads[1] !== nothing
    end

end
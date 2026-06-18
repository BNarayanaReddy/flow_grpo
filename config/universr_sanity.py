import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.run_name = "sanity_test"
    config.debug = False
    config.seed = 42
    config.logdir = "logs"
    config.save_freq = 5
    config.num_checkpoint_limit = 2
    config.mixed_precision = "no"
    
    # Laptop optimization flag
    config.laptop_mode = True
    
    config.dataset = ml_collections.ConfigDict()
    config.dataset.train_file_list = "dataset/sanity_filelist.txt"
    config.dataset.num_samples = 32768
    config.dataset.sr = 48000
    config.dataset.num_workers = 1 if config.laptop_mode else 4

    config.sample = ml_collections.ConfigDict()
    config.sample.num_steps = 1
    config.sample.train_batch_size = 2 # MUST be >= k for DistributedKRepeatSampler
    config.sample.num_image_per_prompt = 2 if config.laptop_mode else 4 # K=2 min for GRPO group advantages
    config.sample.num_batches_per_epoch = 1
    config.sample.noise_level = 0.1
    
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 2 # Matches sample train_batch_size
    config.train.learning_rate = 1e-4
    config.train.adam_beta1 = 0.9
    config.train.adam_beta2 = 0.999
    config.train.adam_weight_decay = 1e-4
    config.train.adam_epsilon = 1e-8
    config.train.gradient_accumulation_steps = 3 if config.laptop_mode else 1
    config.train.max_grad_norm = 1.0
    config.train.clip_eps = 0.2
    config.train.timestep_fraction = 1.0
    config.train.max_workers = 2 if config.laptop_mode else 8

    config.reward_fn = ml_collections.ConfigDict()
    config.reward_fn.lsd = 10.0
    config.reward_fn.openl3 = 1.0

    if config.laptop_mode:
        config.mixed_precision = "fp16"

    config.universr = ml_collections.ConfigDict()
    
    config.universr.model = ml_collections.ConfigDict()
    config.universr.model.in_channels = 2
    config.universr.model.out_channels = 2
    config.universr.model.dims = [16, 32, 64, 128] # Tiny model
    config.universr.model.depths = [1, 1, 1, 1]    # Tiny model
    config.universr.model.drop_path = 0
    config.universr.model.time_dim = 64
    config.universr.model.cond_dim = 128
    config.universr.model.total_freq_bins = 512
    config.universr.model.hr_freq_bins = 432
    config.universr.model.feature_enc_layers = 1
    config.universr.model.cond_dropout_prob = 0.1
    config.universr.model.sr_to_lr_bins = {8: 80, 12: 128, 16: 170, 24: 256}

    config.universr.transform = ml_collections.ConfigDict()
    config.universr.transform.window_fn = 'hann'
    config.universr.transform.n_fft = 1024
    config.universr.transform.sampling_rate = 48000
    config.universr.transform.hop_length = 512
    config.universr.transform.alpha = 0.2
    config.universr.transform.beta = 1
    config.universr.transform.comp_eps = 1.0e-4

    config.universr.path = ml_collections.ConfigDict()
    config.universr.path.init_args = ml_collections.ConfigDict()
    config.universr.path.init_args.sigma_min = 1.0e-4

    return config

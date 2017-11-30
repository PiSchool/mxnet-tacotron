
class Hyperparams:
    '''Hyper parameters'''

    # dataset used
    dataset_name = "bible"

    batch_size = 5
    csv_file = '../train_data/dataset.csv'

    num_epochs = 100
    if dataset_name == "digit_50":
        sr = 8000
        batch_size=5
        sound_fpath = '../digit_50'
        max_seconds_length = 2.9
    if dataset_name == "digit_200":
        sr = 8000
    if dataset_name == "bible":
        batch_size = 15
        sr = 22050
        max_seconds_length = 10
    # data
    csv_file = '../'+dataset_name+'/'+dataset_name+'.csv'
    sound_fpath = '../'+dataset_name

    #max_len = 100 if not sanity_check else 30 # maximum length of text
    #min_len = 10 if not sanity_check else 20 # minimum length of text
    do_text_processing = False

    # signal processing

    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples  This is dependent on the frame_shift.
    win_length = int(sr*frame_length) # samples This is dependent on the frame_length.
    n_mels = 80 # Number of Mel banks to generate
    power = 1.2 # Exponent for amplifying the predicted magnitude
    griffin_lim_iters = 30 # Number of inversion iterations
    ref_level_db = 20
    min_level_db = -100

    # model
    emb_size = 256 # alias = E
    encoder_num_banks = 16
    post_process_num_banks = 8
    num_highwaynet_blocks = 4
    r = 3 # Reduction factor. Paper => 2, 3, 5

    # training scheme
    use_convBank_batchNorm = True
    use_proj1_batchNorm = False
    use_proj2_batchNorm = False
    lr = 0.0005 # Paper => Exponential decay
    #logdir = "logdir" if not sanity_check else "logdir_s"
    #outputdir = 'samples' if not sanity_check else "samples_s"

    #10000 if not sanity_check else 40 # Paper => 2M global steps!
    loss_type = "l2" # Or you can test "l2"
    num_samples = 32

    # NOT USED
    '''
    # etc
    num_gpus = 1 # If you have multiple gpus, adjust this option, and increase the batch size
    	 # and run `train_multiple_gpus.py` instead of `train.py`.
    target_zeros_masking = False # If True, we mask zero padding on the target,
    	                 # so exclude them from the loss calculation.

    # mode
    sanity_check = True
    use_ref_db=False
    normalize = False
    '''

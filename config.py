class FLAGS(object):
    """ """
    """
    General settings
    """
    input_size = 256
    heatmap_size = 32
    cpm_stages = 3
    joint_gaussian_variance = 1.0
    center_radius = 21
    num_of_joints = 21
    color_channel = 'RGB'
    normalize_img = True
    use_gpu = True
    gpu_id = 0


    """
    Demo settings
    """
    # 'MULTI': show multiple stage heatmaps
    # 'SINGLE': show last stage heatmap
    # 'Joint_HM': show last stage heatmap for each joint
    # 'image or video path': show detection on single image or video
    DEMO_TYPE = 'SINGLE'

    model_path = 'cpm_hand'
    cam_id = 0

    webcam_height = 480
    webcam_width = 640

    use_kalman = True
    kalman_noise = 0.03


    """
    Training settings
    """
    network_def = 'cpm_hand'
    train_img_dir = ''
    val_img_dir = ''
    bg_img_dir = ''
    pretrained_model = 'cpm_hand'
    batch_size = 5
    init_lr = 0.001
    lr_decay_rate = 0.5
    lr_decay_step = 10000
    training_iters = 300000
    verbose_iters = 10
    validation_iters = 1000
    model_save_iters = 5000
    augmentation_config = {'hue_shift_limit': (-5, 5),
                           'sat_shift_limit': (-10, 10),
                           'val_shift_limit': (-15, 15),
                           'translation_limit': (-0.15, 0.15),
                           'scale_limit': (-0.3, 0.5),
                           'rotate_limit': (-90, 90)}
    hnm = True  # Make sure generate hnm files first
    do_cropping = True

    """
    For Freeze graphs
    """
    output_node_names = 'stage_3/mid_conv7/BiasAdd:0'


    """
    For Drawing
    """
    # Default Pose
    default_hand = [[259, 335],
                    [245, 311],
                    [226, 288],
                    [206, 270],
                    [195, 261],
                    [203, 308],
                    [165, 290],
                    [139, 287],
                    [119, 284],
                    [199, 328],
                    [156, 318],
                    [128, 314],
                    [104, 318],
                    [204, 341],
                    [163, 340],
                    [133, 347],
                    [108, 349],
                    [206, 359],
                    [176, 368],
                    [164, 370],
                    [144, 377]]

    # Limb connections
    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [0, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [0, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [0, 17],
             [17, 18],
             [18, 19],
             [19, 20]
             ]

    # Finger colors
    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]

    # My hand joint order
    # FLAGS.limbs = [[0, 1],
    #          [1, 2],
    #          [2, 3],
    #          [3, 20],
    #          [4, 5],
    #          [5, 6],
    #          [6, 7],
    #          [7, 20],
    #          [8, 9],
    #          [9, 10],
    #          [10, 11],
    #          [11, 20],
    #          [12, 13],
    #          [13, 14],
    #          [14, 15],
    #          [15, 20],
    #          [16, 17],
    #          [17, 18],
    #          [18, 19],
    #          [19, 20]
    #          ]











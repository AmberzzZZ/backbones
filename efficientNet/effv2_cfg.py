V2S_BLOCKS_ARGS = [   # V2-S
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 24, 'filters_out': 24,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.},
    {'kernel_size': 3, 'repeats': 4, 'filters_in': 24, 'filters_out': 48,
     'expand_ratio': 4, 'id_skip': True, 'strides': 2, 'se_ratio': 0.},
    {'kernel_size': 3, 'repeats': 4, 'filters_in': 48, 'filters_out': 64,
     'expand_ratio': 4, 'id_skip': True, 'strides': 2, 'se_ratio': 0.},
    {'kernel_size': 3, 'repeats': 6, 'filters_in': 64, 'filters_out': 128,
     'expand_ratio': 4, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 9, 'filters_in': 128, 'filters_out': 160,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 15, 'filters_in': 160, 'filters_out': 256,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
]


V2M_BLOCKS_ARGS = [   # V2-M
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 24, 'filters_out': 24,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.},
    {'kernel_size': 3, 'repeats': 5, 'filters_in': 24, 'filters_out': 48,
     'expand_ratio': 4, 'id_skip': True, 'strides': 2, 'se_ratio': 0.},
    {'kernel_size': 3, 'repeats': 5, 'filters_in': 48, 'filters_out': 80,
     'expand_ratio': 4, 'id_skip': True, 'strides': 2, 'se_ratio': 0.},
    {'kernel_size': 3, 'repeats': 7, 'filters_in': 80, 'filters_out': 160,
     'expand_ratio': 4, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 14, 'filters_in': 160, 'filters_out': 176,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 18, 'filters_in': 176, 'filters_out': 304,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 5, 'filters_in': 304, 'filters_out': 512,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
]


V2L_BLOCKS_ARGS = [   # V2-L
    {'kernel_size': 3, 'repeats': 4, 'filters_in': 32, 'filters_out': 32,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.},
    {'kernel_size': 3, 'repeats': 7, 'filters_in': 32, 'filters_out': 64,
     'expand_ratio': 4, 'id_skip': True, 'strides': 2, 'se_ratio': 0.},
    {'kernel_size': 3, 'repeats': 7, 'filters_in': 64, 'filters_out': 96,
     'expand_ratio': 4, 'id_skip': True, 'strides': 2, 'se_ratio': 0.},
    {'kernel_size': 3, 'repeats': 10, 'filters_in': 96, 'filters_out': 192,
     'expand_ratio': 4, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 19, 'filters_in': 196, 'filters_out': 224,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 25, 'filters_in': 224, 'filters_out': 384,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 7, 'filters_in': 384, 'filters_out': 640,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
]





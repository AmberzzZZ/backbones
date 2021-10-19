

hrnet32 = [{'n_blocks': 1, 'n_residuals': 4, 'n_filters': [32,64]},          # stage2
           {'n_blocks': 4, 'n_residuals': 4, 'n_filters': [32,64,128]},      # stage3
           {'n_blocks': 3, 'n_residuals': 4, 'n_filters': [32,64,128,256]},  # stage4
]

hrnet48 = [{'n_blocks': 1, 'n_residuals': 4, 'n_filters': [48,96]},
           {'n_blocks': 4, 'n_residuals': 4, 'n_filters': [48,96,192]},
           {'n_blocks': 3, 'n_residuals': 4, 'n_filters': [48,96,192,384]},
]

hrnet18 = [{'n_blocks': 1, 'n_residuals': 4, 'n_filters': [18,36]},
           {'n_blocks': 4, 'n_residuals': 4, 'n_filters': [18,36,72]},
           {'n_blocks': 3, 'n_residuals': 4, 'n_filters': [18,36,72,144]},
]




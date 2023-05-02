def set_template(args):
    # Set the templates here
    #
    if args.template.find('4X_CGAN') >= 0:
        args.model = 'CGAN_SIMD'
        args.n_resblocks = 20
        args.n_resgroups = 10
        args.patch_size = 192
        args.reduction = 4
        args.scale = "4"

    if args.template.find('8X_CGAN') >= 0:
        args.model = 'CGAN_SIMD'
        args.n_resblocks = 20
        args.n_resgroups = 10
        args.patch_size = 384
        args.reduction = 4
        args.scale = "8"
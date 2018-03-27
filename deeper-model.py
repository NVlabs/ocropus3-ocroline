def make_model(ninput=48, noutput=97):
    B, W, H, D = (0, 900), (0, 9000), ninput, (0, 5000)
    return nn.Sequential(
        # reorder to Torch conventions
        layers.Reorder("BHWD", "BDHW"),
        layers.CheckSizes(B, 1, H, W, name="input"),


        # convolutional layers
        flex.Conv2d(100, 3, padding=(1, 1)), # BDWH
        nn.ReLU(),
        nn.MaxPool2d(3),
        flex.Conv2d(100, 3, padding=(1, 1)), # BDWH
        flex.BatchNorm2d(),
        nn.ReLU(),

        # turn image into sequence
        #layers.Reorder("BDHW", "BWDH"),
        layers.Fun(lambda x: x.view(x.size(0), -1, x.size(3)), "BDHW->BDW"),
        #layers.Reorder("BWD", "BDW"),
        layers.CheckSizes(B, D, W),

        # run 1D LSTM
        flex.Lstm1(100),
        flex.Conv1d(noutput, 1),

        # reorder
        layers.Reorder("BDW", "BWD"),
        layers.CheckSizes(B, W, noutput, name="output")
        )

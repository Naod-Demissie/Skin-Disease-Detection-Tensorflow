from tensorflow.keras.callbacks import Callback

class UnfreezeLayerCallback(Callback):
    def __init__(self, layer_ranges, monitor, base_model_name, mode='auto', patience=3):
        super(UnfreezeLayerCallback, self).__init__()
        self.layer_ranges = layer_ranges
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.wait = 0
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.current_range_index = 0
        self.base_model_name = base_model_name

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.monitor)
        if current_metric is None:
            raise ValueError(f"The specified metric '{self.monitor}' is not found in the training logs.")

        if self.mode == 'auto':
            self.mode = 'min' if 'loss' in self.monitor else 'max'

        if (self.mode == 'min' and current_metric < self.best_metric) or \
            (self.mode == 'max' and current_metric > self.best_metric):
            self.best_metric = current_metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.current_range_index < len(self.layer_ranges):
                    print(f"\nPlateau detected. Unfreezing layers in range: "
                            f"{self.layer_ranges[self.current_range_index]} at epoch {epoch}"
                    )
                self.unfreeze_layers_in_range(self.layer_ranges[self.current_range_index])
                self.current_range_index = min(self.current_range_index + 1, len(self.layer_ranges) - 1)
                self.wait = 0

    def unfreeze_layers_in_range(self, layer_range):
        for layers in self.model.get_layer(self.base_model_name).layers[layer_range]:
            layers.trainable = True
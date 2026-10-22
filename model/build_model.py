from keras import layers, Model


class UCTModel(Model):
    """
    Keras feedforward neural network model for regression tasks (feature extractor and regressor).
    """
    def __init__(self, input_dim, hidden_units=[128, 32], dropout_rate=0.1):
        super().__init__(name="UCTModel")

        self.input_dim = input_dim
        
        # Feature extractor
        self.feature_layers = []
        for units in hidden_units:
            self.feature_layers.append(layers.Dense(units, activation='relu'))

        # Regressor
        self.regressor_layers = [
            layers.Dropout(dropout_rate),
            layers.Dense(1)
        ]

    def build(self, input_shape):
        x_shape = input_shape
        for layer in self.feature_layers:
            layer.build(x_shape)
            x_shape = layer.compute_output_shape(x_shape)
        for layer in self.regressor_layers:
            layer.build(x_shape)
            x_shape = layer.compute_output_shape(x_shape)
        super().build(input_shape)

    def call(self, x, training=False):
        for layer in self.feature_layers:
            x = layer(x, training=training)

        # save latent features 
        latents = x

        for layer in self.regressor_layers:
            x = layer(x, training=training)

        if training:
            return latents, x
        return x

def build_model(input_dim, hidden_units):
    """
    Build and return the UCTModel with the specified input dimension and hidden units.
    """
    model = UCTModel(input_dim=input_dim, hidden_units=hidden_units)
    model.build((None, input_dim))
    model.summary()
    return model
import tensorflow as tf
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.sample_batch import SampleBatch
#from ray.rllib.core.rl_module import RLModule

class KerasModel(TFModelV2):
    """Custom model for PPO."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(KerasModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        self.index = tf.keras.layers.Input(shape=original_space['index'].shape, name="index")
        self.position = tf.keras.layers.Input(shape=original_space['position'].shape, name="position")
        self.macro = tf.keras.layers.Input(shape=original_space['macro'].shape, name="macro")
       
        index = tf.keras.layers.Dense(32, activation='sigmoid')(self.index)
        index = tf.keras.layers.Flatten()(index)
        index = tf.keras.layers.Dense(8, activation='sigmoid')(index)
  
        macro = tf.keras.layers.Dense(8, activation='sigmoid')(self.macro)

        
        concatenated = tf.keras.layers.Concatenate()([index, self.position, macro])

        # Building the dense layers
        layer_out = tf.keras.layers.Dense(num_outputs, activation='sigmoid')(concatenated)
        
        self.value_out = tf.keras.layers.Dense(1, name='value_out')(concatenated)
        
        self.base_model = tf.keras.Model([self.index, self.position, self.macro], [layer_out, self.value_out])
        self.base_model.summary()
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        """Custom core forward method."""
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(input_dict['obs'], self.obs_space, "tf")

        inputs = {'index': orig_obs["index"], 'position': orig_obs["position"], 'macro': orig_obs['macro']}
        model_out, self._value_out = self.base_model(inputs)

        return model_out, state
    
    def value_function(self):
        return tf.reshape(self._value_out, [-1]) 
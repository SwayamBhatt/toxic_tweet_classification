import tensorflow as tf

def save_model(model, filepath):
    """Saves the trained model to a file."""
    model.save(filepath)

def load_model(filepath):
    """Loads a trained model from a file."""
    return tf.keras.models.load_model(filepath)
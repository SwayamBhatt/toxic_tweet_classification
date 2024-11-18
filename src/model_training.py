import tensorflow as tf

def create_vectorizer(vocab_size, sequence_length, train_sentences):
    """Creates and adapts a TextVectorization layer."""
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size, output_sequence_length=sequence_length
    )
    vectorizer.adapt(train_sentences)
    return vectorizer

def create_model(vocab_size, embedding_dim):
    """Creates and compiles the model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
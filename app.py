from src.data_processing import load_data, preprocess_data
from src.model_training import create_vectorizer, create_model
from src.model_evaluation import evaluate_model
from src.model_save_load import save_model

def main():
    # Step 1: Load and preprocess data
    filepath = r"C:\Users\swaya\Desktop\Toxic Tweet Classification\FinalBalancedDataset.csv"
    df = load_data(filepath)
    train_sentences, val_sentences, train_labels, val_labels = preprocess_data(df)

    # Step 2: Create and adapt vectorizer
    vocab_size = 20000
    sequence_length = 10
    vectorizer = create_vectorizer(vocab_size, sequence_length, train_sentences)
    train_sequences = vectorizer(train_sentences)
    val_sequences = vectorizer(val_sentences)

    # Step 3: Create and train model
    embedding_dim = 128
    model = create_model(vocab_size, embedding_dim)
    model.fit(
        train_sequences, train_labels,
        validation_data=(val_sequences, val_labels),
        epochs=5,
        batch_size=32
    )

    # Step 4: Evaluate the model
    evaluate_model(model, val_sequences, val_labels)

    # Step 5: Save the model
    save_model(model, "models/toxic_tweet_model.h5")

if __name__ == "__main__":
    main()
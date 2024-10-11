from scripts.train_model import train_model
from scripts.evaluate_model import evaluate_model

def run_experiments():
    """ Run training and evaluation experiments on the CNN model using spectrogram datasets.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Description
    -----------
    This function sets up the configuration for the experiment, including dataset paths, 
    model save location, number of epochs, and learning rate. It first trains the model using 
    the specified training dataset and then evaluates the trained model on the test dataset. 
    The results of the evaluation are printed to provide the model's performance metrics.
    """
    # Experiment configurations
    dataset_train_path = 'test/data/processed/train_spectrograms'
    dataset_test_path = 'test/data/processed/cwru_spectrograms/2_fault_severity_007'
    model_save_path = 'saved_models/cnn2d.pth'
    num_epochs = 15
    learning_rate = 0.0005
    batch_size=32

    print('Starting model training...')
    train_model(dataset_path=dataset_train_path, model_save_path=model_save_path, 
                num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)

    print('Training completed. Starting model evaluation...')
    evaluate_model(dataset_path=dataset_test_path, model_path=model_save_path)

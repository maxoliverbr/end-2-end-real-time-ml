import os

import turboml as tb
from dotenv import load_dotenv
from loguru import logger

# Establish connection to TurboML platform
load_dotenv()
tb.init(
    backend_url=os.environ['TURBOML_BACKEND_URL'],
    api_key=os.environ['TURBOML_API_KEY']
)

# def get_model(model_name: str) -> tb.Model:


def setup_model(
    model_name: str,
    transactions_dataset_name: str,
    labels_dataset_name: str,
):
    logger.info('Connect to transactions and labels dataset')

    # Get the transactions and labels datasets from TurboML platform
    # Maybe something along these lines
    # I need help with this part :-)
    #   logger.info(f"Connect to transactions dataset {transactions_dataset_name}")
    #   transactions = tb.OnlineDataset.get_model_inputs(id=transactions_dataset_name)
    #   logger.info(f"Connect to labels dataset {labels_dataset_name}")
    #   labels = tb.OnlineDataset.get_model_labels(id=labels_dataset_name)
    breakpoint()
    
    # Define the model
    model = tb.HoeffdingTreeClassifier(n_classes=2)
    # Here you can use play with other models. To check all the available models for
    # Supervised ML, you can use the following command:
    #   tb.ml_algorithms(have_labels=True)
    #
    # Alternatively, you can define your own model in Python
    # https://docs.turboml.com/wyo_models/native_python_model/
    #
    # and even train it on a batch of data, before pushing it to the platform
    # https://docs.turboml.com/wyo_models/batch_python_model/

    numerical_fields = [
        "transactionAmount",
        "localHour",
        "my_sum_feat",
        "my_sql_feat",
    ]
    categorical_fields = [
        "digitalItemCount",
        "physicalItemCount",
        "isProxyIP",
    ]
    features = transactions.get_model_inputs(
        numerical_fields=numerical_fields,
        categorical_fields=categorical_fields
    )
    label = labels.get_model_labels(label_field="is_fraud")

    logger.info(f"Deploy model {model_name} to platform")
    deployed_model_htc = model.deploy(model_name, input=features, labels=label)
    logger.info(f"Deployment of {model_name} completed!")


if __name__ == '__main__':
    
    setup_model(
        model_name='fraud_detection_model',
        transactions_dataset_name="qs_transactions",
        labels_dataset_name="qs_transaction_labels",
    )
import os
import json

import torch
from pathlib import Path
import pytorch_lightning as pl

from utils import parse_input_args
from pl.loggers import TensorBoardLogger

from pl.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
)

from model import LitModel
from datamodule import LitDataModule

from sklearn.metrics import confusion_matrix
import pandas as pd


ml_root = Path("/tmp/output")
num_cpus = os.cpu_count()
EPOCHS = 1


def main(
    trainer_args,
    tensorboard_root="output/tensorboard",
    checkpoint_dir="output/train/models",
    data_path="output/processing",
    # mlpipeline_ui_metadata="mlpipeline-ui-metadata.json",
    # mlpipeline_metrics="mlpipeline-metrics.json",
    # script_args="script_args",
    # confusion_matrix_url="metrics",
    data_module_args=None,
):
    if data_module_args is None:
        data_module_args = {}

    # Callbacks

    lr_logger = LearningRateMonitor()
    tboard = TensorBoardLogger(tensorboard_root, log_graph=True)
    early_stopping = EarlyStopping(
        monitor="val/loss", mode="min", patience=5, verbose=True
    )
    # early_stopping = EarlyStopping(
    #     monitor="val/acc", mode="max", patience=5, verbose=True
    # )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="cifar10_{epoch:02d}",
        filename="epoch={epoch}-val_acc={val/acc:.2f}",
        auto_insert_metric_name=False,
        # save_top_k=1,
        verbose=True,
        monitor="val/acc",
        mode="max",
    )
    model_sum_logger = RichModelSummary(max_depth=-1)

    trainer_add_args = {
        "num_sanity_val_steps": 0,
        "logger": tboard,
        "callbacks": [lr_logger, early_stopping, checkpoint_callback, model_sum_logger],
    }

    # Creating parent directories
    Path(tensorboard_root).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    EPOCHS = trainer_args.get("epochs", 1)

    datamodule = LitDataModule(
        data_path=data_path, num_workers=0, batch_size=16, **data_module_args
    )
    datamodule.setup()

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        log_every_n_steps=1,
        enable_model_summary=False,
        enable_checkpointing=False,
        **trainer_args,
        **trainer_add_args,
    )
    litmodel = LitModel(num_classes=datamodule.num_classes)

    print(":: Training ...")
    trainer.fit(litmodel, datamodule)

    print(":: Saving  Model")
    torch.save(
        litmodel,
    )

    # Load bets checkpoint
    best_model_path = checkpoint_callback.best_model_path
    best_model = LitModel.load_from_checkpoint(best_model_path)

    torch.save(best_model.model.state_dict(), Path("/tmp") / "model.pt")

    # Model Summary
    model_summary_txt = model_sum_logger._summary(trainer, litmodel)
    # print(":: Saving Scripted Model")
    # script = litmodel.to_torchscript()
    # torch.jit.save(script, Path("/tmp") / "model.pt")

    print(":: Testing ...")
    test_results = trainer.test(litmodel, datamodule)
    test_acc = test_results[0]["test/acc"]
    test_loss = test_results[0]["test/loss"]

    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    target_index_list = list(set(litmodel.target))

    num_classes = len(target_index_list)

    class_list = []
    for index in target_index_list:
        class_list.append(classes[index])

    conf_matrix = confusion_matrix(litmodel.target, litmodel.preds)
    confusion_matrix = confusion_matrix.numpy()

    data = []
    for i in range(num_classes):
        for j in range(num_classes):
            data.append({"pred": i, "target": j, "count": conf_matrix[i, j]})

    # Create a DataFrame from the list of dictionaries
    df_cm = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    cm_csv = df_cm.to_csv(header=False, index=False)

    metadata = {
        "outputs": [
            {
                "type": "confusion_matrix",
                "format": "csv",
                "schema": [
                    {"name": "target", "type": "CATEGORY"},
                    {"name": "predicted", "type": "CATEGORY"},
                    {"name": "count", "type": "NUMBER"},
                ],
                "target_col": "actual",
                "predicted_col": "predicted",
                "source": cm_csv,
                "storage": "inline",
                "labels": class_list,
            },
            {
                "storage": "inline",
                "source": """# Model Overview
                                ## Model Summary

                                ```
                                {}
                                ```

                        ## Model Performance

                        **Test Accuracy**: {}
                        **Test Loss**: {}

                        """.format(
                    model_summary_txt,
                    test_acc,
                    test_loss,
                ),
                "type": "markdown",
            },
        ]
    }

    metrics = {
        "metrics": [
            {
                "name": "model_accuracy",
                "numberValue": float(test_acc),
                "format": "PERCENTAGE",
            },
            {
                "name": "model_loss",
                "numberValue": float(test_loss),
                "format": "PERCENTAGE",
            },
        ]
    }

    from collections import namedtuple

    output = namedtuple("output", ["mlpipeline_ui_metadata", "mlpipeline_metrics"])
    return output(json.dumps(metadata), json.dumps(metrics))

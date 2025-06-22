from typing import Dict, Any, Tuple

import torch
from pandas import DataFrame

from freqtrade.freqai.base_models.BasePyTorchRegressor import BasePyTorchRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor, DefaultPyTorchDataConvertor
from freqtrade.freqai.torch.PyTorchLSTMModel import PyTorchLSTMModel
from freqtrade.freqai.torch.PyTorchModelTrainer import PyTorchLSTMTrainer


class PyTorchLSTMRegressor(BasePyTorchRegressor):
    """
    PyTorchLSTMRegressor is a class that uses a PyTorch LSTM model to predict a continuous target variable.

      "model_training_parameters": {
      "learning_rate": 3e-3,
      "trainer_kwargs": {
        "n_steps": null,
        "batch_size": 32,
        "n_epochs": 10,
      },
      "model_kwargs": {
        "num_lstm_layers": 3,
        "hidden_dim": 128,
        "window_size": 5,
        "dropout_percent": 0.4
      }
    }
    """

    @property
    def data_convertor(self) -> PyTorchDataConvertor:
        return DefaultPyTorchDataConvertor(target_tensor_type=torch.float)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        config = self.freqai_info.get("model_training_parameters", {})
        self.learning_rate: float = config.get("learning_rate", 3e-4)
        self.model_kwargs: Dict[str, Any] = config.get("model_kwargs", {})
        self.trainer_kwargs: Dict[str, Any] = config.get("trainer_kwargs", {})
        self.window_size = self.model_kwargs.get('window_size', 10)

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        n_features = data_dictionary["train_features"].shape[-1]
        n_targets = data_dictionary["train_labels"].shape[-1]
        model = PyTorchLSTMModel(input_dim=n_features, output_dim=n_targets, **self.model_kwargs)
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss(reduction='mean')
        trainer = self.get_init_model(dk.pair)
        if trainer is None:
            trainer = PyTorchLSTMTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device,
                data_convertor=self.data_convertor,
                tb_logger=self.tb_logger,
                window_size=self.window_size,
                **self.trainer_kwargs,
            )
        trainer.fit(data_dictionary, self.splits)
        self.model = trainer
        return trainer

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Filter the prediction features data and predict with it.
        :param unfiltered_df: Full dataframe for the current backtest period.
        :return:
        :pred_df: dataframe containing the predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
                     data (NaNs) or felt uncertain about data (PCA and DI index)
        """
        from pandas import DataFrame
        import torch
        
        filtered_df, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )
        filtered_df = dk.feature_pipeline.transform(filtered_df)
        dk.data_dictionary["prediction_features"] = filtered_df

        self.data_convertor.convert_data_for_inference(dk.data_dictionary, dk.pair)
        with torch.no_grad():
            self.model.model.eval()
            y = self.model.model(
                self.data_convertor.x_test
            )
        
        # 处理多目标输出的列名
        if y.shape[1] == len(dk.label_list):
            # 多目标情况：使用所有标签列名
            pred_df = DataFrame(y.detach().tolist(), columns=dk.label_list)
        else:
            # 单目标情况：只使用第一个标签列名
            pred_df = DataFrame(y.detach().tolist(), columns=[dk.label_list[0]])
        
        pred_df, _, _ = dk.label_pipeline.inverse_transform(pred_df)
        return pred_df, dk.do_predict


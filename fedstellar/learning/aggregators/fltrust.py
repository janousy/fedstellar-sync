#
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
#


import logging
import pickle
import copy
from typing import OrderedDict, Optional, List

import torch
from statistics import mean
from fedstellar.learning.aggregators.aggregator import Aggregator
from fedstellar.learning.aggregators.helper import cosine_similarity
from fedstellar.learning.modelmetrics import ModelMetrics
from fedstellar.learning.aggregators.helper import normalise_layers


class FlTrust(Aggregator):
    """
    Federated Averaging (FedAvg) [McMahan et al., 2016]
    Paper: https://arxiv.org/abs/1602.05629
    """

    def __init__(self, node_name="unknown", config=None, logger=None, learner=None, agg_round=0):
        super().__init__(node_name, config, logger, learner, agg_round)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        logging.info("[FLTrust] My config is {}".format(self.config))

    def add_model(self, model: OrderedDict, nodes: List[str], metrics: ModelMetrics):

        super().add_model(model=model, nodes=nodes, metrics=metrics)

    def evaluate_neighbour_model(self, model: OrderedDict, node: str, metrics: ModelMetrics):

        # Cosine Similarity
        model_params = model
        local_params = self.learner.get_parameters()
        cos_similarity: float = cosine_similarity(local_params, model_params)

        # Log model metrics
        if node != self.node_name:
            mapping = {f'cos_{node}': cos_similarity}
            self.learner.logger.log_metrics(metrics=mapping, step=0)

        metrics.cosine_similarity = cos_similarity

        return model, node, metrics

    def aggregate(self, models):
        """
        TrimmedMean [Cao et al., 2022]
        Paper: https://arxiv.org/abs/2012.13995

         Args:
             models: Dictionary with the models (node: model,num_samples).
         """
        # Check if there are models to aggregate
        if len(models) == 0:
            logging.error("[FlTrust] Trying to aggregate models when there is no models")
            return None

        # Compute metrics
        for node_key in models.keys():
            model = models[node_key][0]
            metrics: ModelMetrics = models[node_key][1]
            # the own local model also requires eval to get loss distance
            model_eval, nodes_eval, metrics_eval = self.evaluate_neighbour_model(model, node_key, metrics)
            models[node_key] = (model_eval, metrics_eval)

        # The model of the aggregator serves as a trusted reference
        local_params = models.get(self.node_name)[0]
        if local_params is None:
            logging.error("[FlTrust] Own model as bootstrap is not available")
            return None

        untrusted_models = {k: models[k] for k in models.keys() - {self.node_name}}  # change

        # Normalise the untrusted models
        normalised_models = {}
        for key, msg in untrusted_models.items():
            model_params = msg[0]
            metrics = msg[1]
            if key == self.node_name:
                normalised_models[key] = (local_params, metrics)
            else:
                normalized_params = normalise_layers(model_params, local_params)
                normalised_models[key] = (normalized_params, metrics)

        # Create a Zero Model
        accum = (list(models.values())[-1][0]).copy()
        for layer in accum:
            accum[layer] = torch.zeros_like(accum[layer])

        # Aggregate
        similarities = []
        for client, message in normalised_models.items():
            client_model = message[0]
            metrics: ModelMetrics = message[1]
            similarities.append(metrics.cosine_similarity)
            for layer in client_model:
                accum[layer] = accum[layer] + client_model[layer] * metrics.cosine_similarity

        # Normalize Accum
        sum_similarity = sum(similarities)
        for layer in accum:
            accum[layer] = accum[layer] / sum_similarity

        logging.info("[FlTrust.aggregate] Aggregated model with weights: similarities={}".format(similarities))

        return accum

    def clear(self):
        """
        Clear all for a new aggregation.
        """
        observers = self.get_observers()
        next_round = self.agg_round + 1
        self.__init__(node_name=self.node_name,
                      config=self.config,
                      logger=self.logger,
                      learner=self.learner,
                      agg_round=next_round,
                      )
        for o in observers:
            self.add_observer(o)

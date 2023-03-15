######################################################################
# (C)Copyright 2021 Hewlett Packard Enterprise Development LP
######################################################################

##################################################################
# This file is the main entry point for Swarm Learning for Pytorch
# platform. Users can integrate Swarm framework into their 
# model code by creating an instance of the SwarmCallback class and 
# calling its methods at different phases of training.
##################################################################

from __future__ import print_function
import torch, time
from swarmlearning.pyt import SwarmCallback

# Default Training contract used for learning if not specified by user. 
# Any update to default contract needs similar modifications 
# in all applicable ML platforms (TF, PYT, etc)
DEFAULT_TRAINING_CONTRACT = 'defaultbb.cqdb.sml.hpe'

class Custom_SwarmCallback(SwarmCallback):


    def __init__(self, syncFrequency, minPeers, trainingContract=DEFAULT_TRAINING_CONTRACT, **kwargs):
        '''
        This function initializes the various Swarm network parameters, which 
        are described below -
        :param syncFrequency: Batches of local training to be performed between 
                              2 swarm sync rounds. If adaptive sync enabled, this 
                              is the frequency to be used at the start.
        :param minPeers: Min peers required during each sync round for Swarm to 
                          proceed further.
        :param trainingContract: Training contract associated with this learning. 
                                 Default value is 'defaultbb.cqdb.sml.hpe'.
        :param useAdaptiveSync: Modulate the next interval length post each sync 
                                  round based on perf on validation data.
        :param adsValData: Validation dataset - (X,Y) tuple or generator - used 
                             for adaptive sync
        :param adsValBatch_size: Validation data batch size
        :param checkinModelOnTrainEnd: Indicates which model to check-in once 
                                           local model training ends at a node.
                                           Allowed values: ['inactive', 'snapshot', 
                                           'active']
        :param nodeWeightage: A number between 1-100 to indicate the relative 
                               importance of this node compared to others
        :param mlPlatform: 'Pytorch' ML Platform
        :param model: Pytorch model
        :param logger: Basic Python logger. SwarmCallback class will invoke info, 
                       debug and error methods of this logger to fulfil its need.
                       If no logger is passed, then SwarmCallback class will create 
                       its own logger from basic python logger. If required, user 
                       can get hold of this logger instance to change the log level 
                       as follows -
                       import logging
                       from swarmlearning.pyt import SwarmCallback                       
                       swCallback = SwarmCallback(syncFrequency=128, minPeers=3)
                       swCallback.logger.setLevel(logging.DEBUG)
        '''
        super().__init__(syncFrequency, minPeers, trainingContract, **kwargs)  
        self.sync_done = 0

    def _swarmOnBatchEnd(self):
        '''
        Call SL to periodically merge local params of all models. 
        Should be called to execute swarm functionality at the end
        of each batch of local training
        '''

        self.sync_done = 0
        # If loopback just return
        if self.loopback:
            self.logger.debug("OnBatchEnd: Bypassing Swarm Learning functionality as SWARM_LOOPBACK is True")
            return
        if self.stepsBeforeNextSync == 0 and not self.isSwarmTrainingOver:
            self.logger.debug("="*20 + " swarmOnBatchEnd : START " + "="*20)   
            s = time.time()         
            self._SwarmCallbackBase__doSync()
            e = time.time()
            self.logger.info("Required time to perform swarm weight merging in {} seconds: ".format(e-s))
            self.userMergeDone = True
            self.logger.debug("="*20 + " swarmOnBatchEnd : END " + "="*20)
            self.sync_done = 1
        self.stepsBeforeNextSync -= 1
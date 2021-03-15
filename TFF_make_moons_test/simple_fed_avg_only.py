#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 18:01:14 2021

@author: relogu

Testing a 2-level federated learning algorithm

"""

# importations
import tensorflow as tf
import tensorflow_federated as tff
import nest_asyncio
nest_asyncio.apply()
import attr
from typing import Callable

@tff.federated_computation
def hello_world():
    """Test function."""
    return 'Hello, World!'

hello_world()

#%% classes

@attr.s(eq=False, order=False, frozen=False)
class ServerState(object):
    """Structure for state on the server."""
    
    model = attr.ib()
    
    def _set_model_trainable_weights(self, weights):
        self.model = tff.learning.ModelWeights(trainable=tuple(weights),
                                               non_trainable=tuple())
    

# Convenience type aliases used by google researchers
ModelBuilder = Callable[[], tff.learning.Model]

#%% general functions

def _get_weights(model: tff.learning.Model) -> tff.learning.ModelWeights:
    """Retrieve the weights of the model."""
    return tff.learning.ModelWeights(
        trainable=tuple(model.trainable_variables),
        non_trainable=tuple(model.non_trainable_variables))

#%% server-side functions

@tf.function
def server_update(mean_client_weights):
    """Perform the server model updating step."""
    # creating the new state based on the updated model to be directly returned
    return mean_client_weights
#%% algorithm functions


def build_server_init_fn(weights):
    """Build a `tff.tf_computation` that returns the initial `ServerState`."""

    @tff.tf_computation
    def server_init_tf():
        """Return the initial `ServerState`."""
        return ServerState(model=weights)
    
    return server_init_tf


def build_fed_avg_process(weights) -> tff.templates.IterativeProcess:
    """Call and build all the necessities to build an IterativeProcess."""
    ## preliminary operations for the definition of the fundamental types to 
    ## declare functions and the initialization of the server model
    
    tff.backends.native.set_local_execution_context()
    # initialization of the server (tf_computation)
    server_init_tf = build_server_init_fn(weights)
    
    # retriving important types from the initialized server
    server_state_type = server_init_tf.type_signature.result
    model_weights_type = server_state_type.model
    print(str(server_state_type))
    print(str(model_weights_type))
    
    print("server_update_fn types:")
    @tff.tf_computation(server_state_type)
    def server_update_fn(mean_client_weights):
        """Perform the federated computation of the server update."""
        # return the updated server state
        return server_update(mean_client_weights)
    
    print("run_one_round types:")
    print(str(tff.type_at_server(server_state_type)))
    print(str(tff.type_at_clients(server_state_type)))
    @tff.federated_computation(
        tff.type_at_server(server_state_type),
        tff.type_at_clients(server_state_type))
    def run_one_round(server_state, client_states):
        """Orchestration logic for one round of federated training computation."""
        # performing the federated averaging of the clients' weights
        mean_client_weights = tff.federated_mean(client_states)
        print(str(mean_client_weights))
        ## SERVER UPDATING STEP
        server_state = tff.federated_apply(
            server_update_fn,
            mean_client_weights)

        # returning the new server state
        return server_state
    
    @tff.federated_computation
    def initialize_fn():
        """Initialize the server state."""
        return tff.federated_value(server_init_tf(), tff.SERVER)
    
    # building of the iterative process
    iterative_process = tff.templates.IterativeProcess( # fn that build iter. proc.
        initialize_fn=initialize_fn, # initialization fn for the server state
        next_fn=run_one_round) # the complete fn performing one iterative step
    
    @tff.tf_computation(server_init_tf.type_signature.result)
    def get_model_weights(server_state):
        """Get the server model weights inside the server state."""
        return server_state.model
    
    # assigning the final model weights
    iterative_process.get_model_weights = get_model_weights
    
    # returning the object
    return iterative_process
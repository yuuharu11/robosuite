# src/models/sequence/rnns/cells/ltc.py

import torch
import torch.nn as nn
import numpy as np
from .basic import CellBase
from enum import Enum

class ODESolver(Enum):
    Explicit = "explicit"
    SemiImplicit = "semi_implicit"  
    RungeKutta = "runge_kutta"

class MappingType(Enum):
    Identity = "identity"
    Linear = "linear"
    Affine = "affine"

class LTCCell(CellBase):
    name = 'ltc'  # register the cell name

    def __init__(self, d_input, d_model, solver, ode_solver_unfolds, 
                 input_mapping, erev_init_factor,
                 w_init_max, w_init_min, cm_init_min, cm_init_max,
                 gleak_init_min, gleak_init_max, w_min_value, w_max_value, 
                 gleak_min_value, gleak_max_value, cm_t_min_value, cm_t_max_value,
                 fix_cm, fix_gleak, fix_vleak, **kwargs):
        super().__init__(d_input, d_model, **kwargs)
        
        self._input_size = d_input
        self._num_units = d_model
        self._is_built = False

        # Number of ODE solver steps in one RNN step
        self._ode_solver_unfolds = ode_solver_unfolds
        if solver == "explicit":
            self._solver = ODESolver.Explicit
        elif solver == "runge_kutta":
            self._solver = ODESolver.RungeKutta
        else:
            self._solver = ODESolver.SemiImplicit
        if input_mapping == "identity":
            self._input_mapping = MappingType.Identity
        elif input_mapping == "linear":
            self._input_mapping = MappingType.Linear
        else:
            self._input_mapping = MappingType.Affine

        self._erev_init_factor = erev_init_factor

        self._w_init_max = w_init_max
        self._w_init_min = w_init_min
        self._cm_init_min = cm_init_min
        self._cm_init_max = cm_init_max
        self._gleak_init_min = gleak_init_min
        self._gleak_init_max = gleak_init_max

        self._w_min_value = w_min_value
        self._w_max_value = w_max_value
        self._gleak_min_value = gleak_min_value
        self._gleak_max_value = gleak_max_value
        self._cm_t_min_value = cm_t_min_value
        self._cm_t_max_value = cm_t_max_value

        self._fix_cm = fix_cm
        self._fix_gleak = fix_gleak
        self._fix_vleak = fix_vleak

        # Initialize parameters
        self._get_variables()

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def _map_inputs(self, inputs):
        if self._input_mapping == MappingType.Affine or self._input_mapping == MappingType.Linear:
            inputs = inputs * self.input_w
        if self._input_mapping == MappingType.Affine:
            inputs = inputs + self.input_b
        return inputs

    def forward(self, inputs, state):
        # PyTorchのforwardメソッド (TensorFlowの__call__に相当)
        if not self._is_built:
            self._is_built = True
            
        inputs = self._map_inputs(inputs)

        if self._solver == ODESolver.Explicit:
            next_state = self._ode_step_explicit(inputs, state, self._ode_solver_unfolds)
        elif self._solver == ODESolver.SemiImplicit:
            next_state = self._ode_step(inputs, state)
        elif self._solver == ODESolver.RungeKutta:
            next_state = self._ode_step_runge_kutta(inputs, state)
        else:
            raise ValueError(f"Unknown ODE solver '{str(self._solver)}'")

        outputs = next_state
        
        return outputs, next_state

    # Create PyTorch parameters (TensorFlowの_get_variablesに相当)
    def _get_variables(self):
        # Sensory parameters
        self.register_parameter('sensory_mu', nn.Parameter(torch.empty(self._input_size, self._num_units)))
        self.register_parameter('sensory_sigma', nn.Parameter(torch.empty(self._input_size, self._num_units)))
        self.register_parameter('sensory_W', nn.Parameter(torch.empty(self._input_size, self._num_units)))
        self.register_parameter('sensory_erev', nn.Parameter(torch.empty(self._input_size, self._num_units)))

        # Recurrent parameters
        self.register_parameter('mu', nn.Parameter(torch.empty(self._num_units, self._num_units)))
        self.register_parameter('sigma', nn.Parameter(torch.empty(self._num_units, self._num_units)))
        self.register_parameter('W', nn.Parameter(torch.empty(self._num_units, self._num_units)))
        self.register_parameter('erev', nn.Parameter(torch.empty(self._num_units, self._num_units)))

        # Leak and membrane parameters
        if self._fix_vleak is None:
            self.register_parameter('vleak', nn.Parameter(torch.empty(self._num_units)))
        else:
            self.register_buffer('vleak', torch.full((self._num_units,), self._fix_vleak))

        if self._fix_gleak is None:
            self.register_parameter('gleak', nn.Parameter(torch.empty(self._num_units)))
        else:
            self.register_buffer('gleak', torch.full((self._num_units,), self._fix_gleak))

        if self._fix_cm is None:
            self.register_parameter('cm_t', nn.Parameter(torch.empty(self._num_units)))
        else:
            self.register_buffer('cm_t', torch.full((self._num_units,), self._fix_cm))

        # Input mapping parameters
        if self._input_mapping == MappingType.Affine or self._input_mapping == MappingType.Linear:
            self.register_parameter('input_w', nn.Parameter(torch.ones(self._input_size)))
        if self._input_mapping == MappingType.Affine:
            self.register_parameter('input_b', nn.Parameter(torch.zeros(self._input_size)))

        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        # Sensory parameters initialization
        nn.init.uniform_(self.sensory_mu, 0.3, 0.8)
        nn.init.uniform_(self.sensory_sigma, 3.0, 8.0)
        nn.init.uniform_(self.sensory_W, self._w_init_min, self._w_init_max)
        
        sensory_erev_init = 2 * np.random.randint(0, 2, size=(self._input_size, self._num_units)) - 1
        self.sensory_erev.data.copy_(torch.tensor(sensory_erev_init * self._erev_init_factor, dtype=torch.float32))

        # Recurrent parameters initialization
        nn.init.uniform_(self.mu, 0.3, 0.8)
        nn.init.uniform_(self.sigma, 3.0, 8.0)
        nn.init.uniform_(self.W, self._w_init_min, self._w_init_max)
        
        erev_init = 2 * np.random.randint(0, 2, size=(self._num_units, self._num_units)) - 1
        self.erev.data.copy_(torch.tensor(erev_init * self._erev_init_factor, dtype=torch.float32))

        # Leak and membrane parameters initialization
        if hasattr(self, 'vleak') and isinstance(self.vleak, nn.Parameter):
            nn.init.uniform_(self.vleak, -0.2, 0.2)

        if hasattr(self, 'gleak') and isinstance(self.gleak, nn.Parameter):
            if self._gleak_init_max > self._gleak_init_min:
                nn.init.uniform_(self.gleak, self._gleak_init_min, self._gleak_init_max)
            else:
                nn.init.constant_(self.gleak, self._gleak_init_min)

        if hasattr(self, 'cm_t') and isinstance(self.cm_t, nn.Parameter):
            if self._cm_init_max > self._cm_init_min:
                nn.init.uniform_(self.cm_t, self._cm_init_min, self._cm_init_max)
            else:
                nn.init.constant_(self.cm_t, self._cm_init_min)

    # Hybrid euler method
    def _ode_step(self, inputs, state):
        v_pre = state

        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        sensory_rev_activation = sensory_w_activation * self.sensory_erev

        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        for t in range(self._ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)

            rev_activation = w_activation * self.erev

            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory
            
            numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_numerator
            denominator = self.cm_t + self.gleak + w_denominator

            v_pre = numerator / denominator

        return v_pre

    def _f_prime(self, inputs, state):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        w_reduced_sensory = torch.sum(sensory_w_activation, dim=1)

        # Unfold the multiply ODE multiple times into one RNN step
        w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)

        w_reduced_synapse = torch.sum(w_activation, dim=1)

        sensory_in = self.sensory_erev * sensory_w_activation
        synapse_in = self.erev * w_activation

        sum_in = torch.sum(sensory_in, dim=1) - v_pre * w_reduced_synapse + torch.sum(synapse_in, dim=1) - v_pre * w_reduced_sensory
        
        f_prime = 1 / self.cm_t * (self.gleak * (self.vleak - v_pre) + sum_in)

        return f_prime

    def _ode_step_runge_kutta(self, inputs, state):
        h = 0.1
        for i in range(self._ode_solver_unfolds):
            k1 = h * self._f_prime(inputs, state)
            k2 = h * self._f_prime(inputs, state + k1 * 0.5)
            k3 = h * self._f_prime(inputs, state + k2 * 0.5)
            k4 = h * self._f_prime(inputs, state + k3)

            state = state + 1.0 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return state

    def _ode_step_explicit(self, inputs, state, _ode_solver_unfolds):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        w_reduced_sensory = torch.sum(sensory_w_activation, dim=1)

        # Unfold the multiply ODE multiple times into one RNN step
        for t in range(_ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)

            w_reduced_synapse = torch.sum(w_activation, dim=1)

            sensory_in = self.sensory_erev * sensory_w_activation
            synapse_in = self.erev * w_activation

            sum_in = torch.sum(sensory_in, dim=1) - v_pre * w_reduced_synapse + torch.sum(synapse_in, dim=1) - v_pre * w_reduced_sensory
            
            f_prime = 1 / self.cm_t * (self.gleak * (self.vleak - v_pre) + sum_in)

            v_pre = v_pre + 0.1 * f_prime

        return v_pre
    
    def _sigmoid(self, v_pre, mu, sigma):
        # PyTorch版: メモリ効率を考慮した実装
        v_pre = v_pre.reshape(-1, v_pre.shape[-1], 1)
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def get_param_constrain_op(self):
        """パラメータの制約を適用する (PyTorchでは手動で実行する必要がある)"""
        with torch.no_grad():
            self.cm_t.clamp_(self._cm_t_min_value, self._cm_t_max_value)
            self.gleak.clamp_(self._gleak_min_value, self._gleak_max_value)
            self.W.clamp_(self._w_min_value, self._w_max_value)
            self.sensory_W.clamp_(self._w_min_value, self._w_max_value)

    def step(self, x, state):
        """Single step forward (required by CellBase)"""
        return self.forward(x, state)

    def default_state(self, batch_size, device=None):
        """Create default initial state (required by CellBase)"""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self._num_units, device=device)

    @property
    def d_output(self):
        """Dimension of output (required by CellBase)"""
        return self._num_units

    @property 
    def d_state(self):
        """Dimension of state (required by CellBase)"""
        return self._num_units
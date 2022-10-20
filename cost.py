from typing import Callable, Any

import torch
import qiskit
from ansatz import *


def _comparator(tensor: torch.Tensor,
                state: torch.Tensor,
                use_rows: bool = True) -> torch.Tensor:

    """Calculates the 'probability' of state occurring along
    specified axis in tensor (as frequency)."""

    _occurrences = 0
    if use_rows:
        for row in tensor:
            if torch.all(state == row).item():
                _occurrences += 1
        return _occurrences / tensor.shape[0]
    for col in range(tensor.shape[1]):
        if torch.all(state == tensor[:, col]).item():
            _occurrences += 1
    return torch.tensor([_occurrences / tensor.shape[1]], requires_grad=True)


def _tensor_as_string(tensor: torch.Tensor) -> str:

    """ Transforms 1D pytorch tensor of 0 and 1 to a string."""

    _string = ""
    for value in tensor:
        _string += str(int(value.detach().item()))
    return _string


def NLL_cost(x_train: torch.Tensor,
             x_model: dict,
             eps: float = 1.0e-8,
             use_rows: bool = True) -> float:

    """ Calculates the 'Clipped Negative log-likelihood Loss' """

    # Sample rows of weights (0) or cols of weights (1)
    _my_axis = 1
    if use_rows:
        _my_axis = 0

    # Main loop
    _NR_SHOTS = torch.sum(torch.tensor(list(x_model.values())))
    _cost = torch.zeros(size=(1,), requires_grad=False)
    for idx in range(0, x_train.shape[_my_axis]):
        _P_model = torch.zeros(size=(1,), requires_grad=False)

        if use_rows:
            state = x_train[idx]
            assert len(state) == x_train.shape[1]
        else:
            state = x_train[:, idx]
            assert len(state) == x_train.shape[0]

        if _tensor_as_string(state) in list(x_model.keys()):
            _P_model = _P_model + x_model[_tensor_as_string(state)] / _NR_SHOTS
        _P_train = _comparator(tensor=x_train, state=state, use_rows=use_rows)
        _cost = _cost - _P_train * torch.log(torch.max(torch.tensor([eps, _P_model], requires_grad=False)))

    return _cost


def get_cost(X_train: torch.Tensor,
             _nr_qubits: int = 8,
             _layer_depth: int = 1,
             _shots: int = 512,
             _all_2_all: bool = True,
             _use_rows: bool = True) -> Callable[[Any], float]:

    """Wrapper function to calculate the 'Clipped Negative log-likelihood Loss' on
    the training data (rows or cols of the weight matrix associated w. some layer
    in the discriminator), given som instance of parameters for the rotation gates
    in the Quantum Circuit."""

    backend = qiskit.Aer.get_backend('aer_simulator')

    def execute_circuit(params) -> float:
        q_circuit = ansatz(_theta=params,
                           _nr_qubits=_nr_qubits,
                           _layer_depth=_layer_depth,
                           _all_2_all=_all_2_all,
                           _uniform_warm_start=True)

        transpiled_qpe = qiskit.transpile(q_circuit, backend)
        assembled_qobj = qiskit.assemble(transpiled_qpe, shots=_shots)
        measurements = backend.run(assembled_qobj).result().get_counts()

        return NLL_cost(x_train=X_train,
                        x_model=measurements,
                        eps=1.0e-8,
                        use_rows=_use_rows).item()

    return execute_circuit


def sample_qcirc(params: list[float, ...],
                 _nr_samples: int,
                 _nr_qubits: int,
                 _layer_depth: int = 2,
                 _all_2_all: bool = True,
                 _uniform_warm_start: bool = False) -> torch.Tensor:

    """ Function that samples the Quantum circuit '_nr_samples' times,
    given the params for the rotations gates in the Quantum circuit."""

    backend = qiskit.Aer.get_backend('aer_simulator')
    q_circuit = ansatz(_theta=params,
                       _nr_qubits=_nr_qubits,
                       _layer_depth=_layer_depth,
                       _all_2_all=_all_2_all,
                       _uniform_warm_start=_uniform_warm_start)
    transpiled_qpe = qiskit.transpile(q_circuit, backend)
    assembled_qobj = qiskit.assemble(transpiled_qpe, shots=_nr_samples)
    measurements = backend.run(assembled_qobj).result().get_counts()
    states = []
    for state in list(measurements.keys()):
        _state = []
        for _qbit in state:
            _state.append(float(_qbit))
        for occurrences in range(measurements[state]):
            states.append(_state)
    return torch.tensor(states)


def multibasis_sample_qcirc(params: list[float, ...],
                            _nr_samples: int,
                            _nr_qubits: int,
                            _layer_depth: int = 2,
                            _all_2_all: bool = True,
                            _uniform_warm_start: bool = False) -> torch.Tensor:

    """ Function that samples the Quantum circuit '_nr_samples' times,
    given the params for the rotations gates in the Quantum circuit,
    and concatenates a second measurement of same circuit rotated into
    the orthogonal y-basis."""

    backend = qiskit.Aer.get_backend('aer_simulator')

    # Standard basis
    q_circuit = ansatz(_theta=params,
                       _nr_qubits=_nr_qubits,
                       _layer_depth=_layer_depth,
                       _all_2_all=_all_2_all,
                       _uniform_warm_start=_uniform_warm_start)
    transpiled_qpe = qiskit.transpile(q_circuit, backend)
    assembled_qobj = qiskit.assemble(transpiled_qpe, shots=_nr_samples)
    measurements = backend.run(assembled_qobj).result().get_counts()
    standard_basis_states = []
    for state in list(measurements.keys()):
        _state = []
        for _qbit in state:
            _state.append(float(_qbit))
        for occurrences in range(measurements[state]):
            standard_basis_states.append(_state)
    standard_basis_states = torch.tensor(standard_basis_states)

    # Orthogonal basis
    y_q_circuit = y_basis_ansatz(_theta=params,
                                 _nr_qubits=_nr_qubits,
                                 _layer_depth=_layer_depth,
                                 _all_2_all=_all_2_all,
                                 _uniform_warm_start=_uniform_warm_start)
    y_transpiled_qpe = qiskit.transpile(y_q_circuit, backend)
    y_assembled_qobj = qiskit.assemble(y_transpiled_qpe, shots=_nr_samples)
    y_measurements = backend.run(y_assembled_qobj).result().get_counts()
    y_basis_states = []
    for state in list(y_measurements.keys()):
        _state = []
        for _qbit in state:
            _state.append(float(_qbit))
        for occurrences in range(y_measurements[state]):
            y_basis_states.append(_state)
    y_basis_states = torch.tensor(y_basis_states)

    return torch.cat((standard_basis_states, y_basis_states), dim=1)

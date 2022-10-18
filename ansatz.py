import qiskit


def ansatz(_theta: list[float, ...],
           _nr_qubits: int = 4,
           _layer_depth: int = 2,
           _all_2_all: bool = True,
           _uniform_warm_start: bool = True
           ) -> qiskit.circuit.quantumcircuit.QuantumCircuit:
    if _all_2_all:
        _total_nr_params = _layer_depth * (2 * _nr_qubits + _nr_qubits * (_nr_qubits - 1) / 2)
    else:
        _total_nr_params = _layer_depth * (2 * _nr_qubits + _nr_qubits - 1)
    assert len(
        _theta) == _total_nr_params, f'Theta should contain {_total_nr_params} params, but it contains {len(_theta)}.'

    q_circuit = qiskit.QuantumCircuit(_nr_qubits, _nr_qubits)

    _theta_count = 0

    # Uniform prop. dist. in comp. basis corresponds to hadamard on all
    if _uniform_warm_start:
        for _q in range(_nr_qubits):
            q_circuit.h(_q)

    for _layer in range(_layer_depth):
        for _q in range(_nr_qubits):
            q_circuit.rx(theta=_theta[_theta_count], qubit=_q)
            _theta_count += 1
            q_circuit.rz(phi=_theta[_theta_count], qubit=_q)
            _theta_count += 1

        if _all_2_all:
            # All-2-all entangling
            for _q1 in range(_nr_qubits):
                for _q2 in range(_q1 + 1, _nr_qubits):
                    q_circuit.rxx(theta=_theta[_theta_count], qubit1=_q1, qubit2=_q2)
                    _theta_count += 1
        else:
            # Nearest neighbour entangling
            for _q1 in range(_nr_qubits - 1):
                q_circuit.rxx(theta=_theta[_theta_count], qubit1=_q1, qubit2=_q1 + 1)
                _theta_count += 1

    # Measure
    q_circuit.barrier()
    for _q in range(_nr_qubits):
        q_circuit.measure(_q, _q)

    return q_circuit

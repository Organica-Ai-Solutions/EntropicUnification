#!/usr/bin/env python3
"""
Comprehensive test script for the EntropicUnification engine.

This script tests all core components of the EntropicUnification framework:
- Quantum engine
- Geometry engine
- Entropy module
- Coupling layer
- Loss functions
- Optimizer

It also tests the integration between these components and runs a small simulation
to ensure everything works together properly.
"""

import os
import sys
import math
import time
import argparse
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core modules
from core.quantum_engine import QuantumEngine, QuantumConfig
from core.geometry_engine import GeometryEngine, BoundaryCondition
from core.entropy_module import EntropyModule
from core.coupling_layer import CouplingLayer, StressTensorFormulation
from core.loss_functions import LossFunctions, LossFormulation
from core.optimizer import EntropicOptimizer, OptimizerConfig, OptimizationStrategy, PartitionStrategy

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_quantum_engine():
    """Test the quantum engine."""
    logger.info("Testing quantum engine...")
    
    # Initialize quantum engine
    num_qubits = 2
    depth = 2
    quantum_engine = QuantumEngine(
        config=QuantumConfig(
            num_qubits=num_qubits,
            depth=depth,
            device="default.qubit",
            interface="torch"
        )
    )
    
    # Test state preparation
    # Calculate total parameters needed based on the quantum engine's configuration
    params_per_qubit = quantum_engine.params_per_qubit  # 4: RY, RZ, RX, time evolution
    params_per_entangler = quantum_engine.params_per_entangler  # 2: CRX, CRZ
    total_params = (params_per_qubit * num_qubits + params_per_entangler * (num_qubits - 1)) * depth
    parameters = torch.rand(total_params)
    state = quantum_engine.evolve_state(parameters, time=0.0)
    
    # Check state shape
    expected_shape = (2**num_qubits,)
    assert state.shape == expected_shape, f"State shape mismatch: {state.shape} != {expected_shape}"
    
    # Check state normalization
    norm = torch.abs(torch.sum(torch.abs(state)**2))
    assert abs(norm - 1.0) < 1e-6, f"State not normalized: norm = {norm}"
    
    # Test reduced density matrix
    rho_A = quantum_engine.reduced_density_matrix(state, [0])
    
    # Check density matrix shape
    expected_shape = (2, 2)
    assert rho_A.shape == expected_shape, f"Reduced density matrix shape mismatch: {rho_A.shape} != {expected_shape}"
    
    # Check trace = 1
    trace = torch.trace(rho_A).real
    assert abs(trace - 1.0) < 1e-6, f"Reduced density matrix trace != 1: {trace}"
    
    logger.info("✓ Quantum engine tests passed")
    return True

def test_geometry_engine():
    """Test the geometry engine."""
    logger.info("Testing geometry engine...")
    
    # Initialize geometry engine
    dimensions = 4
    lattice_size = 10
    geometry_engine = GeometryEngine(
        dimensions=dimensions,
        lattice_size=lattice_size,
        boundary_condition=BoundaryCondition.PERIODIC,
        higher_curvature_terms=True,
        alpha_GB=0.1
    )
    
    # Test metric initialization
    assert geometry_engine.metric_field.shape == (lattice_size, dimensions, dimensions), \
        f"Metric field shape mismatch: {geometry_engine.metric_field.shape} != ({lattice_size}, {dimensions}, {dimensions})"
    
    # Test Christoffel symbols
    christoffel = geometry_engine.compute_christoffel_symbols()
    assert christoffel.shape == (lattice_size, dimensions, dimensions, dimensions), \
        f"Christoffel symbols shape mismatch: {christoffel.shape} != ({lattice_size}, {dimensions}, {dimensions}, {dimensions})"
    
    # Test Riemann tensor
    riemann = geometry_engine.compute_riemann_tensor()
    assert riemann.shape == (lattice_size, dimensions, dimensions, dimensions, dimensions), \
        f"Riemann tensor shape mismatch: {riemann.shape} != ({lattice_size}, {dimensions}, {dimensions}, {dimensions}, {dimensions})"
    
    # Test Ricci tensor
    ricci = geometry_engine.compute_ricci_tensor()
    assert ricci.shape == (lattice_size, dimensions, dimensions), \
        f"Ricci tensor shape mismatch: {ricci.shape} != ({lattice_size}, {dimensions}, {dimensions})"
    
    # Test Ricci scalar
    scalar = geometry_engine.compute_ricci_scalar()
    assert scalar.shape == (lattice_size,), \
        f"Ricci scalar shape mismatch: {scalar.shape} != ({lattice_size},)"
    
    # Test Einstein tensor
    einstein = geometry_engine.compute_einstein_tensor()
    assert einstein.shape == (lattice_size, dimensions, dimensions), \
        f"Einstein tensor shape mismatch: {einstein.shape} != ({lattice_size}, {dimensions}, {dimensions})"
    
    # Test metric update
    gradient = torch.rand_like(geometry_engine.metric)
    geometry_engine.update_metric(gradient, learning_rate=0.01)
    
    logger.info("✓ Geometry engine tests passed")
    return True

def test_entropy_module():
    """Test the entropy module."""
    logger.info("Testing entropy module...")
    
    # Initialize quantum engine
    num_qubits = 2
    depth = 2
    quantum_engine = QuantumEngine(
        config=QuantumConfig(
            num_qubits=num_qubits,
            depth=depth,
            device="default.qubit",
            interface="torch"
        )
    )
    
    # Initialize entropy module
    entropy_module = EntropyModule(
        quantum_engine=quantum_engine,
        uv_cutoff=1e-6,
        include_edge_modes=True
    )
    
    # Prepare a Bell state
    bell_state = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.complex128) / np.sqrt(2)
    
    # Test density matrix
    rho = entropy_module.compute_density_matrix(bell_state)
    assert rho.shape == (4, 4), f"Density matrix shape mismatch: {rho.shape} != (4, 4)"
    
    # Test partial trace
    rho_A = entropy_module.partial_trace(bell_state, [0])
    assert rho_A.shape == (2, 2), f"Reduced density matrix shape mismatch: {rho_A.shape} != (2, 2)"
    
    # Test von Neumann entropy
    entropy = entropy_module.von_neumann_entropy(rho_A)
    # Bell state should have entropy close to ln(2) ≈ 0.693
    assert abs(entropy - np.log(2)) < 0.1, f"Entropy value incorrect: {entropy} != {np.log(2)}"
    
    # Test entanglement entropy
    ent_entropy = entropy_module.compute_entanglement_entropy(bell_state, [0])
    # Bell state should have entropy close to ln(2) ≈ 0.693
    assert abs(ent_entropy - np.log(2)) < 0.1, f"Entanglement entropy value incorrect: {ent_entropy} != {np.log(2)}"
    
    # Test entropy gradient
    grad = entropy_module.entropy_gradient(bell_state, [0])
    assert grad.shape == bell_state.shape, f"Entropy gradient shape mismatch: {grad.shape} != {bell_state.shape}"
    
    logger.info("✓ Entropy module tests passed")
    return True

def test_coupling_layer():
    """Test the coupling layer."""
    logger.info("Testing coupling layer...")
    
    # Initialize quantum engine
    num_qubits = 2
    depth = 2
    quantum_engine = QuantumEngine(
        config=QuantumConfig(
            num_qubits=num_qubits,
            depth=depth,
            device="default.qubit",
            interface="torch"
        )
    )
    
    # Initialize geometry engine
    dimensions = 4
    lattice_size = 10
    geometry_engine = GeometryEngine(
        dimensions=dimensions,
        lattice_size=lattice_size,
        boundary_condition=BoundaryCondition.PERIODIC
    )
    
    # Initialize entropy module
    entropy_module = EntropyModule(quantum_engine=quantum_engine)
    
    # Initialize coupling layer
    coupling_layer = CouplingLayer(
        geometry_engine=geometry_engine,
        entropy_module=entropy_module,
        stress_form=StressTensorFormulation.JACOBSON,
        # legacy v1.2 stack under test: opt-in is explicit and local
        allow_legacy_heuristic=True,
    )
    
    # Prepare a Bell state
    bell_state = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.complex128) / np.sqrt(2)
    
    # Test stress tensor computation
    entropy_grad = entropy_module.entropy_gradient(bell_state, [0])
    # legacy v1.2 heuristic: exercised deliberately, with the opt-in that
    # now makes such a call visible at the call site
    stress_tensor, edge_contribution = coupling_layer.compute_entropy_stress_tensor(
        entropy_grad, allow_legacy_heuristic=True)
    assert stress_tensor.shape == (dimensions, dimensions), \
        f"Stress tensor shape mismatch: {stress_tensor.shape} != ({dimensions}, {dimensions})"
    
    # Test Einstein tensor computation
    einstein_tensor, higher_curvature = coupling_layer.compute_einstein_tensor()
    assert einstein_tensor.shape == (dimensions, dimensions), \
        f"Einstein tensor shape mismatch: {einstein_tensor.shape} != ({dimensions}, {dimensions})"
    
    # Test coupling terms computation
    terms = coupling_layer.compute_coupling_terms(bell_state, [0])
    assert terms.entropy_gradient.shape == bell_state.shape, \
        f"Entropy gradient shape mismatch: {terms.entropy_gradient.shape} != {bell_state.shape}"
    assert terms.stress_tensor.shape == (dimensions, dimensions), \
        f"Stress tensor shape mismatch: {terms.stress_tensor.shape} != ({dimensions}, {dimensions})"
    assert terms.einstein_tensor.shape == (dimensions, dimensions), \
        f"Einstein tensor shape mismatch: {terms.einstein_tensor.shape} != ({dimensions}, {dimensions})"
    assert terms.coupling_residual.shape == (dimensions, dimensions), \
        f"Coupling residual shape mismatch: {terms.coupling_residual.shape} != ({dimensions}, {dimensions})"
    
    # Test coupling update
    update_info = coupling_layer.update_coupling(bell_state, [0], learning_rate=0.01)
    assert "consistency" in update_info, "Consistency metric missing from update info"
    
    logger.info("✓ Coupling layer tests passed")
    return True

def test_loss_functions():
    """Test the loss functions."""
    logger.info("Testing loss functions...")
    
    # Initialize quantum engine
    num_qubits = 2
    depth = 2
    quantum_engine = QuantumEngine(
        config=QuantumConfig(
            num_qubits=num_qubits,
            depth=depth,
            device="default.qubit",
            interface="torch"
        )
    )
    
    # Initialize geometry engine
    dimensions = 4
    lattice_size = 10
    geometry_engine = GeometryEngine(
        dimensions=dimensions,
        lattice_size=lattice_size,
        boundary_condition=BoundaryCondition.PERIODIC
    )
    
    # Initialize entropy module
    entropy_module = EntropyModule(quantum_engine=quantum_engine)
    
    # Initialize coupling layer
    coupling_layer = CouplingLayer(
        geometry_engine=geometry_engine,
        entropy_module=entropy_module,
        # legacy v1.2 stack under test: opt-in is explicit and local
        allow_legacy_heuristic=True,
    )
    
    # Initialize loss functions
    loss_functions = LossFunctions(
        coupling_layer=coupling_layer,
        formulation=LossFormulation.STANDARD
    )
    
    # Prepare a Bell state
    bell_state = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.complex128) / np.sqrt(2)
    
    # Compute coupling terms
    terms = coupling_layer.compute_coupling_terms(bell_state, [0])
    
    # Test Einstein constraint loss
    einstein_loss = loss_functions.einstein_constraint_loss(vars(terms))
    assert isinstance(einstein_loss, torch.Tensor), f"Einstein loss is not a tensor: {type(einstein_loss)}"
    
    # Test entropy gradient loss
    target_gradient = torch.zeros_like(terms.entropy_gradient)
    entropy_loss = loss_functions.entropy_gradient_loss(vars(terms), target_gradient)
    assert isinstance(entropy_loss, torch.Tensor), f"Entropy loss is not a tensor: {type(entropy_loss)}"
    
    # Test curvature regularization
    curvature_loss = loss_functions.curvature_regularization()
    assert isinstance(curvature_loss, torch.Tensor), f"Curvature loss is not a tensor: {type(curvature_loss)}"
    
    # Test metric smoothness
    smoothness_loss = loss_functions.metric_smoothness()
    assert isinstance(smoothness_loss, torch.Tensor), f"Smoothness loss is not a tensor: {type(smoothness_loss)}"
    
    # Test total loss
    weights = {"einstein": 1.0, "entropy": 1.0, "curvature": 0.1, "smoothness": 0.1}
    total_loss = loss_functions.total_loss(vars(terms), target_gradient, weights)
    assert "total_loss" in total_loss, "Total loss missing from results"
    
    logger.info("✓ Loss functions tests passed")
    return True

def test_optimizer():
    """Test the optimizer."""
    logger.info("Testing optimizer...")
    
    # Initialize quantum engine
    num_qubits = 2
    depth = 2
    quantum_engine = QuantumEngine(
        config=QuantumConfig(
            num_qubits=num_qubits,
            depth=depth,
            device="default.qubit",
            interface="torch"
        )
    )
    
    # Initialize geometry engine
    dimensions = 4
    lattice_size = 10
    geometry_engine = GeometryEngine(
        dimensions=dimensions,
        lattice_size=lattice_size,
        boundary_condition=BoundaryCondition.PERIODIC
    )
    
    # Initialize entropy module
    entropy_module = EntropyModule(quantum_engine=quantum_engine)
    
    # Initialize coupling layer
    coupling_layer = CouplingLayer(
        geometry_engine=geometry_engine,
        entropy_module=entropy_module,
        # legacy v1.2 stack under test: opt-in is explicit and local
        allow_legacy_heuristic=True,
    )
    
    # Initialize loss functions
    loss_functions = LossFunctions(
        coupling_layer=coupling_layer,
        formulation=LossFormulation.STANDARD
    )
    
    # Initialize optimizer
    optimizer = EntropicOptimizer(
        quantum_engine=quantum_engine,
        geometry_engine=geometry_engine,
        entropy_module=entropy_module,
        coupling_layer=coupling_layer,
        loss_functions=loss_functions,
        config=OptimizerConfig(
            learning_rate=0.01,
            steps=10,
            checkpoint_interval=5,
            log_interval=1,
            results_path="test_results"
        )
    )
    
    # Create parameters and times
    # Calculate total parameters needed based on the quantum engine's configuration
    params_per_qubit = quantum_engine.params_per_qubit  # 4: RY, RZ, RX, time evolution
    params_per_entangler = quantum_engine.params_per_entangler  # 2: CRX, CRZ
    total_params = (params_per_qubit * num_qubits + params_per_entangler * (num_qubits - 1)) * depth
    parameters = torch.rand(total_params)
    times = torch.tensor([0.0])
    partition = [0]  # Trace out qubit 0
    
    # Test optimization step
    state = quantum_engine.evolve_state(parameters, time=0.0)
    target_gradient = torch.zeros_like(entropy_module.entropy_gradient(state, partition))
    weights = {"einstein": 1.0, "entropy": 1.0, "curvature": 0.1, "smoothness": 0.1}
    
    results = optimizer.optimization_step(state, partition, target_gradient, weights)
    assert "total_loss" in results, "Total loss missing from optimization step results"
    
    # Test full training (with reduced steps for testing)
    optimizer.config.steps = 3  # Reduce steps for testing
    final_results = optimizer.train(parameters, times, partition)
    assert "final_state" in final_results, "Final state missing from training results"
    assert "final_metric" in final_results, "Final metric missing from training results"
    assert "history" in final_results, "History missing from training results"
    
    logger.info("✓ Optimizer tests passed")
    return True

def test_integration():
    """Test the integration of all components."""
    logger.info("Testing integration of all components...")
    
    # Initialize quantum engine
    num_qubits = 2
    depth = 2
    quantum_engine = QuantumEngine(
        config=QuantumConfig(
            num_qubits=num_qubits,
            depth=depth,
            device="default.qubit",
            interface="torch"
        )
    )
    
    # Initialize geometry engine
    dimensions = 4
    lattice_size = 10
    geometry_engine = GeometryEngine(
        dimensions=dimensions,
        lattice_size=lattice_size,
        boundary_condition=BoundaryCondition.PERIODIC,
        higher_curvature_terms=True,
        alpha_GB=0.1
    )
    
    # Initialize entropy module
    entropy_module = EntropyModule(
        quantum_engine=quantum_engine,
        uv_cutoff=1e-6,
        include_edge_modes=True
    )
    
    # Initialize coupling layer
    coupling_layer = CouplingLayer(
        geometry_engine=geometry_engine,
        entropy_module=entropy_module,
        stress_form=StressTensorFormulation.JACOBSON,
        include_edge_modes=True,
        include_higher_curvature=True,
        # legacy v1.2 stack under test: opt-in is explicit and local
        allow_legacy_heuristic=True,
    )
    
    # Initialize loss functions
    loss_functions = LossFunctions(
        coupling_layer=coupling_layer,
        formulation=LossFormulation.STANDARD
    )
    
    # Initialize optimizer
    optimizer = EntropicOptimizer(
        quantum_engine=quantum_engine,
        geometry_engine=geometry_engine,
        entropy_module=entropy_module,
        coupling_layer=coupling_layer,
        loss_functions=loss_functions,
        config=OptimizerConfig(
            learning_rate=0.01,
            steps=5,
            checkpoint_interval=5,
            log_interval=1,
            results_path="integration_test_results",
            optimization_strategy=OptimizationStrategy.STANDARD,
            partition_strategy=PartitionStrategy.FIXED
        )
    )
    
    # Create parameters and times
    # Calculate total parameters needed based on the quantum engine's configuration
    params_per_qubit = quantum_engine.params_per_qubit  # 4: RY, RZ, RX, time evolution
    params_per_entangler = quantum_engine.params_per_entangler  # 2: CRX, CRZ
    total_params = (params_per_qubit * num_qubits + params_per_entangler * (num_qubits - 1)) * depth
    parameters = torch.rand(total_params)
    times = torch.tensor([0.0])
    partition = [0]  # Trace out qubit 0
    
    # Run a short training
    final_results = optimizer.train(parameters, times, partition)
    
    # Verify results
    assert "final_state" in final_results, "Final state missing from training results"
    assert "final_metric" in final_results, "Final metric missing from training results"
    assert "history" in final_results, "History missing from training results"
    assert "training_time" in final_results, "Training time missing from training results"
    
    # Check if history contains expected keys
    history = final_results["history"]
    expected_keys = ["total_loss", "einstein_loss", "entropy_loss", "curvature_loss", "smoothness_loss"]
    for key in expected_keys:
        assert key in history, f"History missing key: {key}"
        assert len(history[key]) > 0, f"History key {key} has no entries"
    
    # Check if training converged (loss decreased)
    if len(history["total_loss"]) > 1:
        assert history["total_loss"][-1] <= history["total_loss"][0], "Training did not converge"
    
    logger.info("✓ Integration tests passed")
    return True

def test_validation_gates():
    """Gates must abort on exactly the defects that invalidated v1.2.

    Each assertion here corresponds to a real failure mode:
      - an entropy profile inserted by hand rather than derived (v1.2 defect 1)
      - a wrong contraction breaking tracelessness (v1.2 defect 3)
      - curvature whose identity violations exceed discretization (part of
        v1.2 defect 2 — the symmetry-projection part)

    Scope, honestly: the missing 1/dx normalisation, also part of defect 2, is
    *invisible* to these gates. A uniform scale error multiplies the Riemann
    tensor and its norm equally, so the relative identity violations are
    unchanged. That third of the defect is caught by `test_geometry_engine`'s
    derivative tests, not here.
    """
    from core.validation import (EntropyField, GateConfig, ProvenanceError,
                                 ValidationError, ValidationReport,
                                 check_entropy_field_sanity,
                                 check_entropy_provenance,
                                 check_meaningful_dimension,
                                 check_metric_resolved,
                                 check_riemann_identities, check_tracelessness)

    strict = GateConfig()
    N = 32

    # -- provenance: the v1.2 gate -------------------------------------
    bare = torch.linspace(0.0, 1.0, N, dtype=torch.float64)
    try:
        check_entropy_provenance(bare, strict)
        raise AssertionError("bare tensor accepted as an entropy source")
    except ProvenanceError:
        pass

    inserted = EntropyField.unverified(torch.exp(-((torch.linspace(-3, 3, N)) ** 2)),
                                       "analytic", note="hand-placed Gaussian")
    try:
        check_entropy_provenance(inserted, strict)
        raise AssertionError("hand-placed Gaussian accepted as derived")
    except ProvenanceError:
        pass

    # a field cannot simply claim to be derived
    try:
        EntropyField.unverified(bare, "partial_trace")
        raise AssertionError("unverified() allowed a forged provenance")
    except ValueError:
        pass

    # explicit opt-out still works, and is visible at the call site
    check_entropy_provenance(bare, GateConfig(require_derived_entropy=False))

    # -- provenance cannot be forged by direct construction ---------------
    from core.validation import Provenance
    try:
        EntropyField(bare, Provenance("partial_trace"))
        raise AssertionError("forged derived provenance accepted by constructor")
    except ProvenanceError:
        pass

    # -- and the field is immutable after the claim is recorded -----------
    probe = EntropyField.unverified(bare, "analytic")
    try:
        probe.values = torch.zeros(N, dtype=torch.float64)
        raise AssertionError("EntropyField values were mutable after construction")
    except Exception:
        pass  # frozen dataclass raises FrozenInstanceError

    # -- sanity ---------------------------------------------------------
    relaxed = GateConfig(require_derived_entropy=False)
    for bad, why in (
        (torch.ones(N, dtype=torch.float64), "constant field"),
        (torch.full((N,), -1.0, dtype=torch.float64), "negative entropy"),
    ):
        try:
            check_entropy_field_sanity(EntropyField.unverified(bad), relaxed)
            raise AssertionError(f"{why} accepted")
        except ValidationError:
            pass

    # -- tracelessness ---------------------------------------------------
    g = torch.eye(2, dtype=torch.float64).repeat(N, 1, 1)
    g[:, 0, 0] = -1.0
    traceless = torch.zeros(N, 2, 2, dtype=torch.float64)
    traceless[:, 0, 0] = 1.0
    traceless[:, 1, 1] = 1.0            # g^uv T_uv = -1 + 1 = 0
    assert check_tracelessness(traceless, g, "massless", strict).passed

    not_traceless = torch.zeros(N, 2, 2, dtype=torch.float64)
    not_traceless[:, 1, 1] = 1.0
    try:
        check_tracelessness(not_traceless, g, "massless", strict)
        raise AssertionError("non-traceless MASSLESS tensor accepted")
    except ValidationError:
        pass
    # LAGRANGIAN is not traceless; the gate must not invent a failure
    assert check_tracelessness(not_traceless, g, "lagrangian", strict).passed

    # -- riemann identities: judged against predicted truncation error ----
    # ratio ~1 is what a correct discretization produces at any resolution
    assert check_riemann_identities({"pair_symmetry": 1.4e-3}, strict,
                                    truncation=1e-3).passed
    try:
        # same absolute violation, but the metric is 100x smoother: the
        # discretization cannot explain it, so the curvature code is wrong
        check_riemann_identities({"pair_symmetry": 1.4e-3}, strict,
                                 truncation=1e-5)
        raise AssertionError("violation far exceeding truncation error accepted")
    except ValidationError:
        pass
    # absolute fallback when smoothness is unknown
    try:
        check_riemann_identities({"first_bianchi": 0.5}, strict)
        raise AssertionError("grossly violated Bianchi identity accepted")
    except ValidationError:
        pass

    # -- metric resolution: the gate noise must fail ----------------------
    assert check_metric_resolved(1e-3, strict).passed
    try:
        check_metric_resolved(0.5, strict)
        raise AssertionError("unresolved (noise-scale) metric accepted")
    except ValidationError:
        pass

    # -- dimension --------------------------------------------------------
    # off by default (the framework is 1+1D); fatal when a run claims rigour
    assert check_meaningful_dimension(2, strict).passed
    try:
        check_meaningful_dimension(2, GateConfig(require_meaningful_dimension=True))
        raise AssertionError("2D run accepted as a valid curvature test")
    except ValidationError:
        pass

    # -- end-to-end: a real derived field passes, its raw tensor does not --
    qe = QuantumEngine(config=QuantumConfig(
        num_qubits=4, depth=2, device="default.qubit", interface="torch"))
    em = EntropyModule(qe)
    amps = torch.zeros(16, dtype=torch.complex128)
    amps[0] = amps[-1] = 1.0 / math.sqrt(2.0)          # GHZ
    geo = GeometryEngine(lattice_size=N, dimensions=2)
    r_grid = torch.linspace(1.0, 5.0, N, dtype=torch.float64)
    field = em.entropy_field(amps, torch.linspace(1.5, 2.5, 4).tolist(), r_grid)
    assert field.provenance.derived, "entropy_field() lost its provenance"
    check_entropy_provenance(field, strict)

    # a smooth metric must clear the curvature gates; per-site noise must not
    smooth_geo = GeometryEngine(lattice_size=N, dimensions=2)
    x = torch.linspace(0.0, 1.0, N, dtype=torch.float64)
    with torch.no_grad():
        for a in range(2):
            smooth_geo.metric_field[:, a, a] += 0.1 * torch.sin(2 * math.pi * x)
    smooth_geo._clear_cache()
    assert smooth_geo.validate_curvature().passed, "smooth metric failed gates"

    noisy = GeometryEngine(lattice_size=N, dimensions=2)
    torch.manual_seed(0)
    with torch.no_grad():
        noisy.metric_field += 0.05 * torch.randn_like(noisy.metric_field)
        noisy.metric_field.copy_(
            0.5 * (noisy.metric_field + noisy.metric_field.transpose(1, 2)))
    noisy._clear_cache()
    try:
        noisy.validate_curvature()
        raise AssertionError("metric with no continuum limit passed the gates")
    except ValidationError:
        pass

    cl = CouplingLayer(geo, em)
    report = ValidationReport()
    T = cl.compute_stress_tensor_field(field, formulation="massless", report=report)
    assert report.passed, report.summary()
    assert torch.isfinite(T).all()

    try:
        cl.compute_stress_tensor_field(field.values, formulation="massless")
        raise AssertionError("coupling layer accepted a bare tensor")
    except ProvenanceError:
        pass

    # -- the legacy v1.2 heuristic must refuse to run unasked --------------
    grad = torch.randn(4, dtype=torch.float64)
    try:
        cl.compute_entropy_stress_tensor(grad)   # cl was built without the opt-in
        raise AssertionError("legacy state-space-gradient path ran ungated")
    except ProvenanceError:
        pass
    # per-call opt-in
    cl.compute_entropy_stress_tensor(grad, allow_legacy_heuristic=True)
    # per-object opt-in, declared once at construction
    legacy_cl = CouplingLayer(geo, em, allow_legacy_heuristic=True)
    legacy_cl.compute_entropy_stress_tensor(grad)

    # -- digest binds the claim to the values ------------------------------
    assert field.verify_digest(), "digest did not match its own values"
    forged = EntropyField(torch.zeros_like(field.values), field.provenance,
                          field._token)
    assert not forged.verify_digest(), "substituted values passed the digest"
    try:
        check_entropy_provenance(forged, strict)
        raise AssertionError("substituted values passed the provenance gate")
    except ProvenanceError:
        pass

    # -- gates must actually fire on a metric the optimizer could reach ----
    # (the pre-flight metric is flat, where these gates are near-trivial)
    assert smooth_geo.validate_curvature().passed

    # the report is serialisable so results can ship with their gates
    d = report.to_dict()
    assert d["passed"] and d["gates"], "report did not serialise"

    logger.info("✓ Validation gate tests passed")
    return True


def test_all():
    """Run all tests."""
    logger.info("Starting comprehensive tests of the EntropicUnification engine...")
    
    tests = [
        ("Quantum Engine", test_quantum_engine),
        ("Geometry Engine", test_geometry_engine),
        ("Entropy Module", test_entropy_module),
        ("Coupling Layer", test_coupling_layer),
        ("Loss Functions", test_loss_functions),
        ("Optimizer", test_optimizer),
        ("Validation Gates", test_validation_gates),
        ("Integration", test_integration)
    ]
    
    results = []
    
    for name, test_func in tests:
        logger.info(f"\n{'='*50}\nTesting {name}\n{'='*50}")
        try:
            start_time = time.time()
            success = test_func()
            end_time = time.time()
            
            if success:
                logger.info(f"✓ {name} tests passed in {end_time - start_time:.2f} seconds")
                results.append((name, "PASSED", end_time - start_time))
            else:
                logger.error(f"✗ {name} tests failed")
                results.append((name, "FAILED", end_time - start_time))
        except Exception as e:
            logger.exception(f"✗ {name} tests failed with exception: {e}")
            results.append((name, "ERROR", 0))
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("Test Summary")
    logger.info("="*50)
    
    for name, status, duration in results:
        logger.info(f"{name:20} | {status:10} | {duration:.2f}s")
    
    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status == "FAILED")
    errors = sum(1 for _, status, _ in results if status == "ERROR")
    
    logger.info("-"*50)
    logger.info(f"Passed: {passed}/{len(tests)}, Failed: {failed}, Errors: {errors}")
    
    if failed == 0 and errors == 0:
        logger.info("\n✓ All tests passed!")
        return True
    else:
        logger.error("\n✗ Some tests failed!")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the EntropicUnification engine")
    parser.add_argument("--test", choices=["quantum", "geometry", "entropy", "coupling", "loss", "optimizer", "integration", "all"], 
                        default="all", help="Which component to test")
    args = parser.parse_args()
    
    if args.test == "quantum":
        test_quantum_engine()
    elif args.test == "geometry":
        test_geometry_engine()
    elif args.test == "entropy":
        test_entropy_module()
    elif args.test == "coupling":
        test_coupling_layer()
    elif args.test == "loss":
        test_loss_functions()
    elif args.test == "optimizer":
        test_optimizer()
    elif args.test == "integration":
        test_integration()
    else:
        test_all()

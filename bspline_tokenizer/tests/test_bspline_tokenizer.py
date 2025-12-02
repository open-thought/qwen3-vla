"""Unit tests for the bspline_tokenizer package."""

import numpy as np
import pytest

from bspline_tokenizer import (
    BSplineTokenizer,
    BSplineTrajectory,
    create_clamped_knot_vector,
    bspline_basis_matrix,
)


class TestCreateClampedKnotVector:
    """Tests for create_clamped_knot_vector function."""

    def test_basic_knot_vector(self):
        """Test basic knot vector creation."""
        knots = create_clamped_knot_vector(n_control_points=8, degree=4)
        assert len(knots) == 8 + 4 + 1  # n_cp + degree + 1 = 13

    def test_clamped_structure(self):
        """Test that knot vector has clamped structure."""
        degree = 3
        knots = create_clamped_knot_vector(n_control_points=6, degree=degree)

        # First (degree+1) knots should be 0
        assert np.all(knots[: degree + 1] == 0.0)
        # Last (degree+1) knots should be 1
        assert np.all(knots[-(degree + 1) :] == 1.0)

    def test_knot_vector_monotonic(self):
        """Test that knot vector is non-decreasing."""
        knots = create_clamped_knot_vector(n_control_points=10, degree=4)
        assert np.all(np.diff(knots) >= 0)

    def test_memoization(self):
        """Test that function returns cached results."""
        knots1 = create_clamped_knot_vector(8, 4)
        knots2 = create_clamped_knot_vector(8, 4)
        # Should return the exact same array object (memoized)
        assert knots1 is knots2

    def test_different_params_different_results(self):
        """Test that different parameters give different results."""
        knots1 = create_clamped_knot_vector(8, 4)
        knots2 = create_clamped_knot_vector(8, 3)
        knots3 = create_clamped_knot_vector(10, 4)
        assert len(knots1) != len(knots2)
        assert len(knots1) != len(knots3)


class TestBSplineBasisMatrix:
    """Tests for bspline_basis_matrix function."""

    def test_partition_of_unity(self):
        """Test that basis functions sum to 1 at any point."""
        t_values = np.linspace(0, 1, 50)
        basis = bspline_basis_matrix(t_values, n_control_points=8, degree=4)
        row_sums = np.sum(basis, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_non_negative(self):
        """Test that basis functions are non-negative."""
        t_values = np.linspace(0, 1, 50)
        basis = bspline_basis_matrix(t_values, n_control_points=8, degree=4)
        assert np.all(basis >= 0)

    def test_output_shape(self):
        """Test that output has correct shape."""
        t_values = np.linspace(0, 1, 100)
        n_cp = 12
        basis = bspline_basis_matrix(t_values, n_control_points=n_cp, degree=4)
        assert basis.shape == (100, n_cp)

    def test_endpoint_interpolation(self):
        """Test that endpoints use only first/last control point."""
        n_cp = 8
        degree = 4
        basis = bspline_basis_matrix(np.array([0.0, 1.0]), n_control_points=n_cp, degree=degree)

        # At t=0, only first basis function should be 1
        np.testing.assert_allclose(basis[0, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(basis[0, 1:], 0.0, atol=1e-10)

        # At t=1, only last basis function should be 1
        np.testing.assert_allclose(basis[1, -1], 1.0, atol=1e-10)
        np.testing.assert_allclose(basis[1, :-1], 0.0, atol=1e-10)


class TestBSplineTrajectory:
    """Tests for BSplineTrajectory class."""

    def test_init_from_control_points(self):
        """Test initialization from control points."""
        control_points = np.random.randn(7, 8)
        traj = BSplineTrajectory(control_points, degree=4)

        assert traj.n_dof == 7
        assert traj.n_control_points == 8
        assert traj.degree == 4
        np.testing.assert_array_equal(traj.control_points, control_points)

    def test_init_1d_control_points(self):
        """Test that 1D control points are reshaped correctly."""
        control_points = np.random.randn(8)
        traj = BSplineTrajectory(control_points, degree=4)

        assert traj.n_dof == 1
        assert traj.n_control_points == 8
        assert traj.control_points.shape == (1, 8)

    def test_evaluate_single_point(self):
        """Test evaluation at a single point."""
        control_points = np.random.randn(3, 8)
        traj = BSplineTrajectory(control_points, degree=4)

        result = traj.evaluate(0.5)
        assert result.shape == (3,)

    def test_evaluate_multiple_points(self):
        """Test evaluation at multiple points."""
        control_points = np.random.randn(3, 8)
        traj = BSplineTrajectory(control_points, degree=4)

        t = np.linspace(0, 1, 50)
        result = traj.evaluate(t)
        assert result.shape == (50, 3)

    def test_callable_interface(self):
        """Test __call__ interface."""
        control_points = np.random.randn(3, 8)
        traj = BSplineTrajectory(control_points, degree=4)

        result1 = traj(0.5)
        result2 = traj.evaluate(0.5)
        np.testing.assert_array_equal(result1, result2)

    def test_endpoint_interpolation(self):
        """Test that trajectory passes through first and last control points."""
        control_points = np.array([[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]])
        traj = BSplineTrajectory(control_points, degree=4)

        # At t=0, should equal first control point
        np.testing.assert_allclose(traj(0.0), control_points[:, 0], atol=1e-10)
        # At t=1, should equal last control point
        np.testing.assert_allclose(traj(1.0), control_points[:, -1], atol=1e-10)

    def test_evaluate_out_of_range(self):
        """Test that evaluation outside [0,1] raises error."""
        control_points = np.random.randn(3, 8)
        traj = BSplineTrajectory(control_points, degree=4)

        with pytest.raises(ValueError, match="Time values must be in"):
            traj.evaluate(-0.1)

        with pytest.raises(ValueError, match="Time values must be in"):
            traj.evaluate(1.1)

    def test_fit_basic(self):
        """Test fitting a trajectory to data."""
        t = np.linspace(0, 1, 100)
        trajectory = np.sin(2 * np.pi * t).reshape(-1, 1)

        traj = BSplineTrajectory.fit(t, trajectory, n_control_points=8, degree=4)

        assert traj.n_dof == 1
        assert traj.n_control_points == 8

    def test_fit_multi_dof(self):
        """Test fitting a multi-DoF trajectory."""
        t = np.linspace(0, 1, 100)
        trajectory = np.column_stack(
            [np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), t**2]
        )

        traj = BSplineTrajectory.fit(t, trajectory, n_control_points=10, degree=4)

        assert traj.n_dof == 3
        assert traj.n_control_points == 10

    def test_fit_with_bounds(self):
        """Test fitting with bounds on control points."""
        t = np.linspace(0, 1, 100)
        trajectory = 2 * np.sin(2 * np.pi * t).reshape(-1, 1)  # Range roughly [-2, 2]

        traj = BSplineTrajectory.fit(
            t, trajectory, n_control_points=8, degree=4, bounds=(-1.5, 1.5)
        )

        # Control points should be within bounds
        assert np.all(traj.control_points >= -1.5)
        assert np.all(traj.control_points <= 1.5)

    def test_fit_reconstruction_quality(self):
        """Test that fitted trajectory reconstructs original well."""
        t = np.linspace(0, 1, 100)
        trajectory = np.sin(2 * np.pi * t).reshape(-1, 1)

        traj = BSplineTrajectory.fit(t, trajectory, n_control_points=12, degree=4)
        reconstructed = traj.evaluate(t)

        # Should reconstruct with low error
        mae = np.mean(np.abs(reconstructed - trajectory))
        assert mae < 0.01

    def test_fit_invalid_params(self):
        """Test that invalid parameters raise errors."""
        t = np.linspace(0, 1, 100)
        trajectory = np.random.randn(100, 3)

        with pytest.raises(ValueError, match="n_control_points.*must be >= degree"):
            BSplineTrajectory.fit(t, trajectory, n_control_points=3, degree=4)


class TestBSplineTokenizer:
    """Tests for BSplineTokenizer class."""

    @pytest.fixture
    def tokenizer(self):
        """Create a default tokenizer for tests."""
        return BSplineTokenizer(
            n_dof=7,
            n_control_points=8,
            degree=4,
            bounds=(-1.5, 1.5),
            n_bins=255,
            token_order="basis_first",
        )

    @pytest.fixture
    def sample_trajectory(self):
        """Create a sample trajectory for tests."""
        np.random.seed(42)
        t = np.linspace(0, 1, 50)
        trajectory = np.zeros((50, 7))
        for dof in range(7):
            freq = np.random.uniform(0.5, 2)
            trajectory[:, dof] = 0.5 * np.sin(2 * np.pi * freq * t)
        return t, trajectory

    def test_init(self, tokenizer):
        """Test tokenizer initialization."""
        assert tokenizer.n_dof == 7
        assert tokenizer.n_control_points == 8
        assert tokenizer.degree == 4
        assert tokenizer.bounds == (-1.5, 1.5)
        assert tokenizer.n_bins == 255
        assert tokenizer.vocab_size == 255
        assert tokenizer.n_tokens == 56  # 7 * 8

    def test_init_invalid_params(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="n_control_points.*must be >= degree"):
            BSplineTokenizer(n_control_points=3, degree=4)

        with pytest.raises(ValueError, match="token_order must be"):
            BSplineTokenizer(token_order="invalid")

    def test_encode_shape(self, tokenizer, sample_trajectory):
        """Test that encode returns correct shape."""
        t, trajectory = sample_trajectory
        tokens = tokenizer.encode(t, trajectory)

        assert tokens.shape == (tokenizer.n_tokens,)
        assert tokens.dtype == np.int32

    def test_encode_token_range(self, tokenizer, sample_trajectory):
        """Test that tokens are within valid range."""
        t, trajectory = sample_trajectory
        tokens = tokenizer.encode(t, trajectory)

        assert np.all(tokens >= 0)
        assert np.all(tokens < tokenizer.n_bins)

    def test_decode_returns_trajectory(self, tokenizer, sample_trajectory):
        """Test that decode returns BSplineTrajectory."""
        t, trajectory = sample_trajectory
        tokens = tokenizer.encode(t, trajectory)
        decoded = tokenizer.decode(tokens)

        assert isinstance(decoded, BSplineTrajectory)
        assert decoded.n_dof == tokenizer.n_dof
        assert decoded.n_control_points == tokenizer.n_control_points

    def test_encode_decode_roundtrip(self, tokenizer, sample_trajectory):
        """Test encode-decode roundtrip reconstruction."""
        t, trajectory = sample_trajectory
        tokens = tokenizer.encode(t, trajectory)
        decoded = tokenizer.decode(tokens)
        reconstructed = decoded.evaluate(t)

        # Should reconstruct reasonably well (allowing for quantization)
        mae = np.mean(np.abs(reconstructed - trajectory))
        assert mae < 0.05  # Allow some quantization error

    def test_null_trajectory_roundtrip(self, tokenizer):
        """Test that null trajectory (all zeros) roundtrips perfectly."""
        t = np.linspace(0, 1, 50)
        trajectory = np.zeros((50, tokenizer.n_dof))

        tokens = tokenizer.encode(t, trajectory)
        decoded = tokenizer.decode(tokens)
        reconstructed = decoded.evaluate(t)

        # With 255 bins (odd) and symmetric bounds, zero should be exactly representable
        np.testing.assert_allclose(reconstructed, trajectory, atol=1e-10)

    def test_roundtrip_511_bins(self):
        """Test encode-decode roundtrip with 511 bins."""
        tokenizer = BSplineTokenizer(n_dof=7, n_bins=511)
        np.random.seed(123)
        t = np.linspace(0, 1, 50)
        trajectory = np.zeros((50, 7))
        for dof in range(7):
            freq = np.random.uniform(0.5, 2)
            trajectory[:, dof] = 0.5 * np.sin(2 * np.pi * freq * t)

        tokens = tokenizer.encode(t, trajectory)
        decoded = tokenizer.decode(tokens)
        reconstructed = decoded.evaluate(t)

        mae = np.mean(np.abs(reconstructed - trajectory))
        assert mae < 0.05

    def test_roundtrip_512_bins(self):
        """Test encode-decode roundtrip with 512 bins."""
        tokenizer = BSplineTokenizer(n_dof=7, n_bins=512)
        np.random.seed(123)
        t = np.linspace(0, 1, 50)
        trajectory = np.zeros((50, 7))
        for dof in range(7):
            freq = np.random.uniform(0.5, 2)
            trajectory[:, dof] = 0.5 * np.sin(2 * np.pi * freq * t)

        tokens = tokenizer.encode(t, trajectory)
        decoded = tokenizer.decode(tokens)
        reconstructed = decoded.evaluate(t)

        mae = np.mean(np.abs(reconstructed - trajectory))
        assert mae < 0.05

    def test_decode_invalid_length(self, tokenizer):
        """Test that decode raises error for wrong token length."""
        wrong_tokens = np.zeros(10, dtype=np.int32)
        with pytest.raises(ValueError, match="Expected .* tokens"):
            tokenizer.decode(wrong_tokens)

    def test_token_order_basis_first(self):
        """Test basis_first token ordering."""
        tokenizer = BSplineTokenizer(
            n_dof=3, n_control_points=6, degree=3, token_order="basis_first"
        )
        # In basis_first: [cp0_j0, cp0_j1, cp0_j2, cp1_j0, cp1_j1, cp1_j2, ...]
        # Use values within bounds (-1.5, 1.5)
        control_points = (np.arange(18).reshape(3, 6).astype(float) - 9) * 0.15
        tokens = tokenizer.get_tokens_from_control_points(control_points)

        recovered = tokenizer.get_control_points_from_tokens(tokens)
        np.testing.assert_allclose(recovered, control_points, atol=0.02)

    def test_token_order_joint_first(self):
        """Test joint_first token ordering."""
        tokenizer = BSplineTokenizer(
            n_dof=3, n_control_points=6, degree=3, token_order="joint_first"
        )
        # In joint_first: [cp0_j0, cp1_j0, cp2_j0, cp3_j0, cp0_j1, ...]
        # Use values within bounds (-1.5, 1.5)
        control_points = (np.arange(18).reshape(3, 6).astype(float) - 9) * 0.15
        tokens = tokenizer.get_tokens_from_control_points(control_points)

        recovered = tokenizer.get_control_points_from_tokens(tokens)
        np.testing.assert_allclose(recovered, control_points, atol=0.02)

    def test_quantization_dequantization(self, tokenizer):
        """Test quantization and dequantization are inverses."""
        values = np.linspace(-1.5, 1.5, 100)
        tokens = tokenizer._quantize(values)
        recovered = tokenizer._dequantize(tokens)

        # Should be close but with quantization error
        max_error = (tokenizer.bounds[1] - tokenizer.bounds[0]) / (
            tokenizer.n_bins - 1
        )
        np.testing.assert_allclose(recovered, values, atol=max_error / 2 + 1e-10)

    def test_quantization_clipping(self, tokenizer):
        """Test that out-of-bounds values are clipped."""
        values = np.array([-10.0, 0.0, 10.0])
        tokens = tokenizer._quantize(values)

        assert tokens[0] == 0  # Clipped to min
        assert tokens[2] == tokenizer.n_bins - 1  # Clipped to max

    def test_compute_reconstruction_error(self, tokenizer, sample_trajectory):
        """Test reconstruction error computation."""
        t, trajectory = sample_trajectory
        errors = tokenizer.compute_reconstruction_error(t, trajectory)

        assert "mae" in errors
        assert "max_error" in errors
        assert "rmse" in errors
        assert "mae_per_dof" in errors
        assert "max_error_per_dof" in errors

        assert errors["mae"] >= 0
        assert errors["max_error"] >= errors["mae"]
        assert len(errors["mae_per_dof"]) == tokenizer.n_dof

    def test_repr(self, tokenizer):
        """Test string representation."""
        repr_str = repr(tokenizer)
        assert "BSplineTokenizer" in repr_str
        assert "n_dof=7" in repr_str
        assert "n_control_points=8" in repr_str


class TestTokenizeTrajectoryFunction:
    """Tests for the convenience tokenize_trajectory function."""

    def test_basic_usage(self):
        """Test basic usage of tokenize_trajectory."""
        from bspline_tokenizer import tokenize_trajectory

        t = np.linspace(0, 1, 50)
        trajectory = np.sin(2 * np.pi * t).reshape(-1, 1)

        tokens, tokenizer = tokenize_trajectory(t, trajectory)

        assert isinstance(tokens, np.ndarray)
        assert isinstance(tokenizer, BSplineTokenizer)
        assert tokenizer.n_dof == 1

    def test_infers_n_dof(self):
        """Test that n_dof is inferred from trajectory."""
        from bspline_tokenizer import tokenize_trajectory

        t = np.linspace(0, 1, 50)
        trajectory = np.random.randn(50, 5)

        tokens, tokenizer = tokenize_trajectory(t, trajectory)

        assert tokenizer.n_dof == 5

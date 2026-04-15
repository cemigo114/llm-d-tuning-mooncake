package scorer

import "math"

// KalmanFilter1D implements a univariate Kalman filter for tracking
// a single system metric (e.g., queue variance, KV utilization spread).
//
// Unlike EMA which applies a fixed smoothing factor, the Kalman filter
// adapts its gain based on the relative confidence between the prediction
// model and the measurement. This produces better estimates during
// transient load changes (e.g., burst traffic) while remaining stable
// during steady state.
//
// State model:
//   x(k) = x(k-1) + w, w ~ N(0, Q)   (process noise — system variability)
//   z(k) = x(k) + v,   v ~ N(0, R)    (measurement noise — metric jitter)
//
// The filter maintains:
//   - x̂: estimated state (the "true" metric value)
//   - P: estimation uncertainty
//   - K: Kalman gain (how much to trust new measurements vs prediction)
type KalmanFilter1D struct {
	// State estimate
	X float64
	// Estimation uncertainty (error covariance)
	P float64
	// Process noise variance — how much the true state changes per step.
	// Higher Q = faster response to changes, more noise in estimate.
	Q float64
	// Measurement noise variance — how noisy the raw metric readings are.
	// Higher R = smoother estimate, slower response to changes.
	R float64
	// Whether the filter has been initialized with a measurement.
	initialized bool
}

// NewKalmanFilter1D creates a Kalman filter with specified noise parameters.
//
// Tuning guide:
//   - Q/R ratio determines responsiveness vs smoothness
//   - Q/R > 1: responsive, tracks changes quickly (good for bursty metrics)
//   - Q/R < 1: smooth, filters out noise (good for stable metrics)
//   - Q/R = 1: balanced
//
// For queue variance: Q=1.0, R=2.0 (moderately smooth — queues change fast)
// For KV utilization spread: Q=0.5, R=3.0 (smoother — utilization is stable)
// For cache hit rate: Q=0.1, R=5.0 (very smooth — hit rate is a slow signal)
func NewKalmanFilter1D(processNoise, measurementNoise float64) *KalmanFilter1D {
	return &KalmanFilter1D{
		Q: processNoise,
		R: measurementNoise,
		P: 1.0, // Initial uncertainty — will converge quickly
	}
}

// Update performs a predict-then-update cycle with a new measurement.
// Returns the updated state estimate.
//
// The Kalman gain K determines the weighting:
//   K close to 1.0 → trust the measurement (high process noise or low meas noise)
//   K close to 0.0 → trust the prediction (low process noise or high meas noise)
func (kf *KalmanFilter1D) Update(measurement float64) float64 {
	if !kf.initialized {
		kf.X = measurement
		kf.P = 1.0
		kf.initialized = true
		return kf.X
	}

	// Predict step (state transition is identity — we assume steady state)
	// x̂⁻ = x̂ (prediction = previous estimate)
	// P⁻ = P + Q (uncertainty grows by process noise)
	predictedP := kf.P + kf.Q

	// Update step
	// K = P⁻ / (P⁻ + R) — Kalman gain
	K := predictedP / (predictedP + kf.R)

	// x̂ = x̂⁻ + K * (z - x̂⁻) — correct prediction with measurement
	innovation := measurement - kf.X
	kf.X = kf.X + K*innovation

	// P = (1 - K) * P⁻ — update uncertainty
	kf.P = (1 - K) * predictedP

	return kf.X
}

// Gain returns the current Kalman gain (0-1).
// Useful for observability — a high gain means the filter is adapting quickly.
func (kf *KalmanFilter1D) Gain() float64 {
	if !kf.initialized {
		return 1.0
	}
	predictedP := kf.P + kf.Q
	return predictedP / (predictedP + kf.R)
}

// Steady returns true if the filter has converged to a steady gain.
// Convergence is defined as the gain changing by less than 1% per step.
func (kf *KalmanFilter1D) Steady() bool {
	if !kf.initialized {
		return false
	}
	// At steady state, P converges to: P_ss = (-Q + sqrt(Q^2 + 4*Q*R)) / 2
	pSteady := (-kf.Q + math.Sqrt(kf.Q*kf.Q+4*kf.Q*kf.R)) / 2
	return math.Abs(kf.P-pSteady)/pSteady < 0.01
}

// KalmanMetricTracker wraps three Kalman filters for the adaptive scorer's
// three tracked metrics: queue variance, KV utilization spread, and cache hit rate.
type KalmanMetricTracker struct {
	QueueVariance *KalmanFilter1D
	KVUtilSpread  *KalmanFilter1D
	CacheHitRate  *KalmanFilter1D
}

// NewKalmanMetricTracker creates a tracker with tuned noise parameters
// for each metric type.
func NewKalmanMetricTracker() *KalmanMetricTracker {
	return &KalmanMetricTracker{
		// Queue variance: responsive (queues change fast under load)
		QueueVariance: NewKalmanFilter1D(1.0, 2.0),
		// KV utilization spread: moderately smooth
		KVUtilSpread: NewKalmanFilter1D(0.5, 3.0),
		// Cache hit rate: very smooth (slow-moving signal)
		CacheHitRate: NewKalmanFilter1D(0.1, 5.0),
	}
}

// Update all three filters with new measurements and return filtered values.
func (t *KalmanMetricTracker) Update(queueVar, kvSpread, cacheHitRate float64) (
	filteredQueueVar, filteredKVSpread, filteredCacheHit float64,
) {
	return t.QueueVariance.Update(queueVar),
		t.KVUtilSpread.Update(kvSpread),
		t.CacheHitRate.Update(cacheHitRate)
}

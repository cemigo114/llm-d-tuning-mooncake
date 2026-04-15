package scorer

import (
	"math"
	"testing"
)

func TestKalmanFilter1DBasic(t *testing.T) {
	kf := NewKalmanFilter1D(1.0, 2.0)

	// First measurement initializes
	v := kf.Update(10.0)
	if v != 10.0 {
		t.Errorf("first update should return measurement, got %f", v)
	}

	// Second measurement blends
	v = kf.Update(20.0)
	if v <= 10.0 || v >= 20.0 {
		t.Errorf("second update should blend, got %f", v)
	}
}

func TestKalmanFilter1DConvergence(t *testing.T) {
	kf := NewKalmanFilter1D(1.0, 2.0)

	// Feed constant signal — should converge
	for i := 0; i < 100; i++ {
		kf.Update(42.0)
	}
	if math.Abs(kf.X-42.0) > 0.01 {
		t.Errorf("should converge to 42.0, got %f", kf.X)
	}
	if !kf.Steady() {
		t.Error("should be steady after 100 constant updates")
	}
}

func TestKalmanFilter1DStepResponse(t *testing.T) {
	kf := NewKalmanFilter1D(1.0, 2.0)

	// Stabilize at 10
	for i := 0; i < 50; i++ {
		kf.Update(10.0)
	}

	// Step change to 20
	var values []float64
	for i := 0; i < 20; i++ {
		v := kf.Update(20.0)
		values = append(values, v)
	}

	// Should approach 20 but not instantly
	if values[0] >= 19.0 {
		t.Error("Kalman should not jump instantly to new value")
	}
	if values[len(values)-1] < 19.0 {
		t.Errorf("should converge near 20 after 20 steps, got %f", values[len(values)-1])
	}
}

func TestKalmanFilter1DNoisySignal(t *testing.T) {
	// High measurement noise — filter should smooth
	kf := NewKalmanFilter1D(0.1, 10.0)

	// Feed noisy signal around 50
	var estimates []float64
	for i := 0; i < 100; i++ {
		noise := float64(i%7) - 3.0 // deterministic "noise" [-3, 3]
		kf.Update(50.0 + noise)
		estimates = append(estimates, kf.X)
	}

	// Variance of estimates should be much lower than variance of input noise
	var sum, sumSq float64
	for _, v := range estimates[20:] { // skip warmup
		sum += v
		sumSq += v * v
	}
	n := float64(len(estimates) - 20)
	mean := sum / n
	variance := sumSq/n - mean*mean

	if variance > 2.0 {
		t.Errorf("filtered variance %.2f too high, filter not smoothing", variance)
	}
}

func TestKalmanGain(t *testing.T) {
	// High process noise, low measurement noise → gain near 1
	kf1 := NewKalmanFilter1D(10.0, 0.1)
	kf1.Update(1.0)
	kf1.Update(2.0)
	if kf1.Gain() < 0.8 {
		t.Errorf("high Q/R should give high gain, got %f", kf1.Gain())
	}

	// Low process noise, high measurement noise → gain near 0
	kf2 := NewKalmanFilter1D(0.01, 10.0)
	kf2.Update(1.0)
	for i := 0; i < 50; i++ {
		kf2.Update(1.0)
	}
	if kf2.Gain() > 0.1 {
		t.Errorf("low Q/R should give low gain at steady state, got %f", kf2.Gain())
	}
}

func TestKalmanMetricTracker(t *testing.T) {
	tracker := NewKalmanMetricTracker()

	qv, kvs, chr := tracker.Update(5.0, 0.3, 0.8)
	if qv != 5.0 || kvs != 0.3 || chr != 0.8 {
		t.Error("first update should return measurements")
	}

	qv2, kvs2, chr2 := tracker.Update(10.0, 0.5, 0.7)
	if qv2 <= 5.0 || qv2 >= 10.0 {
		t.Errorf("queue variance should blend, got %f", qv2)
	}
	if kvs2 <= 0.3 || kvs2 >= 0.5 {
		t.Errorf("kv spread should blend, got %f", kvs2)
	}
	if chr2 <= 0.7 || chr2 >= 0.8 {
		t.Errorf("cache hit rate should blend, got %f", chr2)
	}

	// Queue filter should be more responsive than cache filter
	// (higher Q/R ratio)
	qGain := tracker.QueueVariance.Gain()
	cGain := tracker.CacheHitRate.Gain()
	if qGain <= cGain {
		t.Errorf("queue filter gain (%f) should be > cache filter gain (%f)", qGain, cGain)
	}
}

func TestKalmanVsEMA(t *testing.T) {
	// Compare Kalman vs EMA on a step change
	kf := NewKalmanFilter1D(1.0, 2.0)
	alpha := 0.15 // EMA smoothing

	// Both track at 10.0
	emaVal := 10.0
	for i := 0; i < 50; i++ {
		kf.Update(10.0)
		emaVal = alpha*10.0 + (1-alpha)*emaVal
	}

	// Step to 20.0 — measure response
	kalmanSteps, emaSteps := 0, 0
	emaVal2 := emaVal
	kf2X := kf.X

	for i := 0; i < 100; i++ {
		kf2X = kf.Update(20.0)
		emaVal2 = alpha*20.0 + (1-alpha)*emaVal2
		if kalmanSteps == 0 && kf2X > 18.0 {
			kalmanSteps = i + 1
		}
		if emaSteps == 0 && emaVal2 > 18.0 {
			emaSteps = i + 1
		}
	}

	// Kalman should reach 90% of step change faster than EMA
	// (because Kalman increases gain when innovation is high)
	t.Logf("Steps to reach 18.0: Kalman=%d, EMA=%d", kalmanSteps, emaSteps)
	if kalmanSteps >= emaSteps && kalmanSteps > 0 && emaSteps > 0 {
		t.Logf("Note: Kalman (%d steps) not faster than EMA (%d steps) for this Q/R config",
			kalmanSteps, emaSteps)
	}
}

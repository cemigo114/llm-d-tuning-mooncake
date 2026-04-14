package scorer

import (
	"context"
	"encoding/json"
	"math"
	"testing"

	k8stypes "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func newAdaptiveTestEndpoint(name string, waitingQueue int, kvCachePercent float64) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: name, Namespace: "default"}},
		&fwkdl.Metrics{
			WaitingQueueSize:    waitingQueue,
			RunningRequestsSize: 0,
			KVCacheUsagePercent: kvCachePercent,
		},
		nil,
	)
}

func TestVariance(t *testing.T) {
	tests := []struct {
		name   string
		values []float64
		want   float64
	}{
		{"empty", nil, 0},
		{"single", []float64{5}, 0},
		{"uniform", []float64{3, 3, 3}, 0},
		{"varied", []float64{0, 10}, 25},
		{"small variance", []float64{1, 2, 3}, 0.6666666666666666},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := variance(tt.values)
			if math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("variance(%v) = %f, want %f", tt.values, got, tt.want)
			}
		})
	}
}

func TestSpread(t *testing.T) {
	tests := []struct {
		name   string
		values []float64
		want   float64
	}{
		{"empty", nil, 0},
		{"single", []float64{5}, 0},
		{"range", []float64{1, 5, 3}, 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := spread(tt.values)
			if got != tt.want {
				t.Errorf("spread(%v) = %f, want %f", tt.values, got, tt.want)
			}
		})
	}
}

func TestClamp(t *testing.T) {
	if clamp(5, 0, 10) != 5 {
		t.Error("clamp should pass through in-range values")
	}
	if clamp(-1, 0, 10) != 0 {
		t.Error("clamp should enforce lower bound")
	}
	if clamp(15, 0, 10) != 10 {
		t.Error("clamp should enforce upper bound")
	}
}

func TestAdaptiveScorerFactory(t *testing.T) {
	params := AdaptiveParameters{
		Alpha:                  0.1,
		QueueVarianceThreshold: 5.0,
		KVUtilSpreadThreshold:  0.4,
		CacheHitRateThreshold:  0.1,
		QueueDeferThreshold:    3.0,
		DeferDuration:          "5ms",
		PrefixBaseWeight:       4.0,
		LoadBaseWeight:         2.0,
		ActiveBaseWeight:       1.0,
	}
	raw, _ := json.Marshal(params)

	p, err := AdaptiveScorerFactory("test-adaptive", raw, nil)
	if err != nil {
		t.Fatalf("factory failed: %v", err)
	}

	scorer, ok := p.(*AdaptiveScorer)
	if !ok {
		t.Fatal("factory did not return *AdaptiveScorer")
	}

	if scorer.TypedName().Type != AdaptiveScorerType {
		t.Errorf("type = %s, want %s", scorer.TypedName().Type, AdaptiveScorerType)
	}
	if scorer.Category() != scheduling.Distribution {
		t.Error("category should be Distribution")
	}
}

func TestAdaptiveScorerDefaults(t *testing.T) {
	p, err := AdaptiveScorerFactory("default-test", nil, nil)
	if err != nil {
		t.Fatalf("factory with nil params failed: %v", err)
	}

	scorer := p.(*AdaptiveScorer)
	if scorer.params.Alpha != defaultAlpha {
		t.Errorf("alpha = %f, want %f", scorer.params.Alpha, defaultAlpha)
	}
	if scorer.params.PrefixBaseWeight != defaultPrefixBaseWeight {
		t.Errorf("prefixBaseWeight = %f, want %f", scorer.params.PrefixBaseWeight, defaultPrefixBaseWeight)
	}
}

func TestAdaptiveScorerEmptyEndpoints(t *testing.T) {
	p, _ := AdaptiveScorerFactory("empty-test", nil, nil)
	scorer := p.(*AdaptiveScorer)

	result := scorer.Score(context.Background(), nil, nil, nil)
	if result != nil {
		t.Error("Score with nil endpoints should return nil")
	}
}

func TestAdaptiveScorerScoring(t *testing.T) {
	p, _ := AdaptiveScorerFactory("score-test", nil, nil)
	scorer := p.(*AdaptiveScorer)

	endpoints := []scheduling.Endpoint{
		newAdaptiveTestEndpoint("pod-idle", 0, 50.0),
		newAdaptiveTestEndpoint("pod-busy", 10, 80.0),
	}

	scores := scorer.Score(context.Background(), nil, nil, endpoints)
	if len(scores) != 2 {
		t.Fatalf("expected 2 scores, got %d", len(scores))
	}

	// Idle pod should score higher than busy pod (lower queue = higher load score)
	idleScore := scores[endpoints[0]]
	busyScore := scores[endpoints[1]]
	if idleScore <= busyScore {
		t.Errorf("idle pod (%f) should score higher than busy pod (%f)", idleScore, busyScore)
	}
}

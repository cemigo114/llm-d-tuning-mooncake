package scorer

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const (
	AdaptiveScorerType = "adaptive-scorer"

	defaultAlpha = 0.15

	defaultQueueVarianceHigh   = 4.0
	defaultKVUtilSpreadHigh    = 0.3
	defaultCacheHitRateLow     = 0.2
	defaultQueueDeferThreshold = 4.0 // raised from 2.0 based on A/B results

	defaultPrefixBaseWeight = 3.0
	defaultLoadBaseWeight   = 2.0
	defaultActiveBaseWeight = 2.0

	minWeight = 0.5
	maxWeight = 10.0

	// Workload classification thresholds.
	longContextPromptChars = 20000 // ~5K tokens at 4 chars/token
)

// workloadProfile defines per-workload weight presets and defer behavior.
type workloadProfile struct {
	name           string
	prefixWeight   float64
	loadWeight     float64
	activeWeight   float64
	deferThreshold float64 // override QueueDeferThreshold per profile
}

var (
	// conversationalProfile: high prefix sharing (system prompts, multi-turn).
	// Favor prefix cache affinity, moderate defer threshold.
	conversationalProfile = workloadProfile{
		name: "conversational", prefixWeight: 4.0, loadWeight: 2.0, activeWeight: 1.5,
		deferThreshold: 3.0,
	}
	// diverseProfile: low prefix sharing (tool-calling, varied prompts).
	// Favor load balance over prefix, higher defer threshold to avoid over-deferring.
	diverseProfile = workloadProfile{
		name: "diverse", prefixWeight: 1.5, loadWeight: 3.0, activeWeight: 2.5,
		deferThreshold: 6.0,
	}
	// longContextProfile: very long input sequences (>5K tokens).
	// Aggressive load balance to avoid queueing behind slow prefills.
	longContextProfile = workloadProfile{
		name: "long-context", prefixWeight: 2.0, loadWeight: 4.0, activeWeight: 3.0,
		deferThreshold: 2.0,
	}
	// defaultProfile: baseline when no classification signal is available.
	defaultAdaptiveProfile = workloadProfile{
		name: "default", prefixWeight: 3.0, loadWeight: 2.0, activeWeight: 2.0,
		deferThreshold: 4.0,
	}
)

// AdaptiveParameters configures the adaptive scorer behavior.
type AdaptiveParameters struct {
	Alpha                  float64 `json:"alpha"`
	QueueVarianceThreshold float64 `json:"queueVarianceThreshold"`
	KVUtilSpreadThreshold  float64 `json:"kvUtilSpreadThreshold"`
	CacheHitRateThreshold  float64 `json:"cacheHitRateThreshold"`
	QueueDeferThreshold    float64 `json:"queueDeferThreshold"`
	DeferDuration          string  `json:"deferDuration"`
	PrefixBaseWeight       float64 `json:"prefixBaseWeight"`
	LoadBaseWeight         float64 `json:"loadBaseWeight"`
	ActiveBaseWeight       float64 `json:"activeBaseWeight"`

	// EnableWorkloadProfiles activates per-request workload classification.
	// When true, base weights are overridden by the detected profile.
	EnableWorkloadProfiles bool `json:"enableWorkloadProfiles"`
}

// adaptiveState tracks runtime metrics via EMA for weight adjustment.
type adaptiveState struct {
	mu sync.RWMutex

	queueVarianceEMA float64
	kvUtilSpreadEMA  float64
	cacheHitRateEMA  float64

	prefixWeight float64
	loadWeight   float64
	activeWeight float64

	totalRequests uint64
	deferCount    uint64

	// Per-profile counters for observability.
	profileCounts map[string]uint64
}

var _ scheduling.Scorer = &AdaptiveScorer{}

func AdaptiveScorerFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	params := AdaptiveParameters{
		Alpha:                  defaultAlpha,
		QueueVarianceThreshold: defaultQueueVarianceHigh,
		KVUtilSpreadThreshold:  defaultKVUtilSpreadHigh,
		CacheHitRateThreshold:  defaultCacheHitRateLow,
		QueueDeferThreshold:    defaultQueueDeferThreshold,
		DeferDuration:          "10ms",
		PrefixBaseWeight:       defaultPrefixBaseWeight,
		LoadBaseWeight:         defaultLoadBaseWeight,
		ActiveBaseWeight:       defaultActiveBaseWeight,
		EnableWorkloadProfiles: true,
	}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &params); err != nil {
			return nil, fmt.Errorf("failed to parse adaptive-scorer parameters: %w", err)
		}
	}

	deferDur, err := time.ParseDuration(params.DeferDuration)
	if err != nil {
		deferDur = 10 * time.Millisecond
	}

	s := &AdaptiveScorer{
		typedName: plugin.TypedName{Type: AdaptiveScorerType, Name: name},
		params:    params,
		deferDur:  deferDur,
		state: &adaptiveState{
			prefixWeight:  params.PrefixBaseWeight,
			loadWeight:    params.LoadBaseWeight,
			activeWeight:  params.ActiveBaseWeight,
			profileCounts: make(map[string]uint64),
		},
	}
	return s, nil
}

type AdaptiveScorer struct {
	typedName plugin.TypedName
	params    AdaptiveParameters
	deferDur  time.Duration
	state     *adaptiveState
}

func (s *AdaptiveScorer) TypedName() plugin.TypedName { return s.typedName }
func (s *AdaptiveScorer) WithName(name string) *AdaptiveScorer {
	s.typedName.Name = name
	return s
}
func (s *AdaptiveScorer) Category() scheduling.ScorerCategory { return scheduling.Distribution }

// classifyWorkload determines the workload profile for a request.
// Classification uses prompt length and request type as signals.
func (s *AdaptiveScorer) classifyWorkload(request *scheduling.LLMRequest) workloadProfile {
	if !s.params.EnableWorkloadProfiles || request == nil || request.Body == nil {
		return defaultAdaptiveProfile
	}

	// Check prompt length for long-context classification.
	promptLen := 0
	isChatCompletion := false

	if request.Body.ChatCompletions != nil {
		isChatCompletion = true
		for _, msg := range request.Body.ChatCompletions.Messages {
			promptLen += len(msg.Content.Raw)
			for _, block := range msg.Content.Structured {
				promptLen += len(block.Text)
			}
		}
	}

	if request.Body.Completions != nil && request.Body.Completions.Prompt != "" {
		promptLen = len(request.Body.Completions.Prompt)
	}

	// Also use RequestSizeBytes as a fallback signal.
	if promptLen == 0 && request.RequestSizeBytes > 0 {
		promptLen = request.RequestSizeBytes
	}

	// Classification hierarchy: long-context > conversational > diverse.
	if promptLen > longContextPromptChars {
		return longContextProfile
	}
	if isChatCompletion {
		return conversationalProfile
	}
	// Short completions without chat structure = likely tool/agent pattern.
	if promptLen > 0 {
		return diverseProfile
	}

	return defaultAdaptiveProfile
}

func (s *AdaptiveScorer) Score(ctx context.Context, _ *scheduling.CycleState,
	request *scheduling.LLMRequest, endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {

	logger := log.FromContext(ctx).V(logutil.DEFAULT)

	if len(endpoints) == 0 {
		return nil
	}

	// ---- Workload classification ----
	profile := s.classifyWorkload(request)

	s.state.mu.Lock()
	s.state.profileCounts[profile.name]++
	s.state.mu.Unlock()

	// ---- Observe system state ----
	queueDepths := make([]float64, 0, len(endpoints))
	kvUtils := make([]float64, 0, len(endpoints))
	var minQueue float64 = math.MaxFloat64

	for _, ep := range endpoints {
		m := ep.GetMetrics()
		qd := float64(m.WaitingQueueSize)
		queueDepths = append(queueDepths, qd)
		if qd < minQueue {
			minQueue = qd
		}
		kvUtil := float64(m.KVCacheUsagePercent) / 100.0
		kvUtils = append(kvUtils, kvUtil)
	}

	queueVar := variance(queueDepths)
	kvSpread := spread(kvUtils)

	// ---- Queue threshold deferral (profile-aware) ----
	deferThreshold := profile.deferThreshold
	if deferThreshold > 0 && minQueue > deferThreshold {
		s.state.mu.Lock()
		s.state.deferCount++
		s.state.mu.Unlock()
		logger.Info("Adaptive scorer: deferring",
			"profile", profile.name, "minQueue", minQueue,
			"threshold", deferThreshold)
		time.Sleep(s.deferDur)
	}

	// ---- EMA update + profile-aware weight adjustment ----
	s.state.mu.Lock()
	alpha := s.params.Alpha
	s.state.queueVarianceEMA = alpha*queueVar + (1-alpha)*s.state.queueVarianceEMA
	s.state.kvUtilSpreadEMA = alpha*kvSpread + (1-alpha)*s.state.kvUtilSpreadEMA
	s.state.totalRequests++

	// Start from profile base weights instead of global defaults.
	prefixW := profile.prefixWeight
	loadW := profile.loadWeight
	activeW := profile.activeWeight

	// Dynamic adjustments on top of profile weights.
	if s.state.queueVarianceEMA > s.params.QueueVarianceThreshold {
		boostFactor := 1.0 + (s.state.queueVarianceEMA-s.params.QueueVarianceThreshold)/s.params.QueueVarianceThreshold
		loadW *= boostFactor
		activeW *= boostFactor
		prefixW *= 0.7
	}

	if s.state.kvUtilSpreadEMA > s.params.KVUtilSpreadThreshold {
		loadW *= 1.3
	}

	if s.state.cacheHitRateEMA < s.params.CacheHitRateThreshold && s.state.totalRequests > 100 {
		prefixW *= 0.5
	}

	prefixW = clamp(prefixW, minWeight, maxWeight)
	loadW = clamp(loadW, minWeight, maxWeight)
	activeW = clamp(activeW, minWeight, maxWeight)

	totalW := prefixW + loadW + activeW
	prefixW /= totalW
	loadW /= totalW
	activeW /= totalW

	s.state.prefixWeight = prefixW
	s.state.loadWeight = loadW
	s.state.activeWeight = activeW
	s.state.mu.Unlock()

	// ---- Composite scores ----
	scoredEndpoints := make(map[scheduling.Endpoint]float64, len(endpoints))

	for i, ep := range endpoints {
		m := ep.GetMetrics()

		prefixScore := float64(m.KVCacheUsagePercent) / 100.0

		loadScore := 0.5
		if queueDepths[i] == 0 {
			loadScore = 1.0
		} else {
			loadScore = math.Max(0, 0.5*(1.0-queueDepths[i]/128.0))
		}

		activeScore := 1.0
		running := float64(m.RunningRequestsSize)
		if running > 0 {
			activeScore = math.Max(0, 1.0-running/64.0)
		}

		composite := prefixW*prefixScore + loadW*loadScore + activeW*activeScore
		scoredEndpoints[ep] = composite
	}

	if s.state.totalRequests%100 == 0 {
		s.state.mu.RLock()
		logger.Info("Adaptive scorer state",
			"profile", profile.name,
			"prefix", fmt.Sprintf("%.3f", prefixW),
			"load", fmt.Sprintf("%.3f", loadW),
			"active", fmt.Sprintf("%.3f", activeW),
			"queueVarEMA", fmt.Sprintf("%.2f", s.state.queueVarianceEMA),
			"deferCount", s.state.deferCount,
			"profiles", s.state.profileCounts)
		s.state.mu.RUnlock()
	}

	return scoredEndpoints
}

func variance(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	var sum float64
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))
	var sumSq float64
	for _, v := range values {
		d := v - mean
		sumSq += d * d
	}
	return sumSq / float64(len(values))
}

func spread(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	mn, mx := values[0], values[0]
	for _, v := range values[1:] {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	return mx - mn
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

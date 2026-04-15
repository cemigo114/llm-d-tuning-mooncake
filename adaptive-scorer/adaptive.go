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

	defaultQueueVarianceHigh   = 4.0
	defaultKVUtilSpreadHigh    = 0.3
	defaultCacheHitRateLow     = 0.2
	defaultQueueDeferThreshold = 4.0

	defaultPrefixBaseWeight = 3.0
	defaultLoadBaseWeight   = 2.0
	defaultActiveBaseWeight = 2.0

	minWeight = 0.5
	maxWeight = 10.0

	longContextPromptChars = 20000

	// Kalman filter noise defaults.
	// These control the responsiveness vs smoothness tradeoff.
	// Q/R > 1: responsive (tracks changes fast)
	// Q/R < 1: smooth (filters noise)
	defaultQueueProcessNoise = 1.0 // queues change fast
	defaultQueueMeasureNoise = 2.0
	defaultKVProcessNoise    = 0.5 // KV util is moderately stable
	defaultKVMeasureNoise    = 3.0
	defaultCacheProcessNoise = 0.1 // cache hit rate is slow-moving
	defaultCacheMeasureNoise = 5.0

	// Online adaptation defaults.
	defaultOnlineDecayWindow = 200
	defaultTTFTThreshold     = 0.5 // 500ms — endpoints above this get penalized
)

type workloadProfile struct {
	name           string
	prefixWeight   float64
	loadWeight     float64
	activeWeight   float64
	deferThreshold float64
}

var (
	conversationalProfile = workloadProfile{
		name: "conversational", prefixWeight: 4.0, loadWeight: 2.0, activeWeight: 1.5,
		deferThreshold: 3.0,
	}
	diverseProfile = workloadProfile{
		name: "diverse", prefixWeight: 1.5, loadWeight: 3.0, activeWeight: 2.5,
		deferThreshold: 6.0,
	}
	longContextProfile = workloadProfile{
		name: "long-context", prefixWeight: 2.0, loadWeight: 4.0, activeWeight: 3.0,
		deferThreshold: 2.0,
	}
	defaultAdaptiveProfile = workloadProfile{
		name: "default", prefixWeight: 3.0, loadWeight: 2.0, activeWeight: 2.0,
		deferThreshold: 4.0,
	}
)

// AdaptiveParameters configures the adaptive scorer.
type AdaptiveParameters struct {
	// Kalman filter noise (replaces EMA alpha).
	QueueProcessNoise float64 `json:"queueProcessNoise"`
	QueueMeasureNoise float64 `json:"queueMeasureNoise"`
	KVProcessNoise    float64 `json:"kvProcessNoise"`
	KVMeasureNoise    float64 `json:"kvMeasureNoise"`
	CacheProcessNoise float64 `json:"cacheProcessNoise"`
	CacheMeasureNoise float64 `json:"cacheMeasureNoise"`

	// Weight adjustment thresholds.
	QueueVarianceThreshold float64 `json:"queueVarianceThreshold"`
	KVUtilSpreadThreshold  float64 `json:"kvUtilSpreadThreshold"`
	CacheHitRateThreshold  float64 `json:"cacheHitRateThreshold"`
	QueueDeferThreshold    float64 `json:"queueDeferThreshold"`
	DeferDuration          string  `json:"deferDuration"`

	// Base weights.
	PrefixBaseWeight float64 `json:"prefixBaseWeight"`
	LoadBaseWeight   float64 `json:"loadBaseWeight"`
	ActiveBaseWeight float64 `json:"activeBaseWeight"`

	// Feature flags.
	EnableWorkloadProfiles bool `json:"enableWorkloadProfiles"`
	EnableOnlineAdaptation bool `json:"enableOnlineAdaptation"`

	// Online adaptation: window size for TTFT tracking per endpoint.
	OnlineDecayWindow int     `json:"onlineDecayWindow"`
	TTFTThreshold     float64 `json:"ttftThreshold"`
}

// endpointPerf tracks per-endpoint TTFT for online adaptation.
type endpointPerf struct {
	ring     []float64
	writeIdx int
	count    int
}

func newEndpointPerf(window int) *endpointPerf {
	return &endpointPerf{ring: make([]float64, window)}
}

func (ep *endpointPerf) record(ttft float64) {
	ep.ring[ep.writeIdx] = ttft
	ep.writeIdx = (ep.writeIdx + 1) % len(ep.ring)
	if ep.count < len(ep.ring) {
		ep.count++
	}
}

func (ep *endpointPerf) meanTTFT() float64 {
	if ep.count == 0 {
		return 0
	}
	var sum float64
	for i := 0; i < ep.count; i++ {
		sum += ep.ring[i]
	}
	return sum / float64(ep.count)
}

type adaptiveState struct {
	mu sync.RWMutex

	kalman *KalmanMetricTracker

	prefixWeight float64
	loadWeight   float64
	activeWeight float64

	totalRequests uint64
	deferCount    uint64
	profileCounts map[string]uint64

	// Online adaptation state.
	endpointPerfs map[string]*endpointPerf
	perfWindow    int
}

var _ scheduling.Scorer = &AdaptiveScorer{}

func AdaptiveScorerFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	params := AdaptiveParameters{
		QueueProcessNoise:      defaultQueueProcessNoise,
		QueueMeasureNoise:      defaultQueueMeasureNoise,
		KVProcessNoise:         defaultKVProcessNoise,
		KVMeasureNoise:         defaultKVMeasureNoise,
		CacheProcessNoise:      defaultCacheProcessNoise,
		CacheMeasureNoise:      defaultCacheMeasureNoise,
		QueueVarianceThreshold: defaultQueueVarianceHigh,
		KVUtilSpreadThreshold:  defaultKVUtilSpreadHigh,
		CacheHitRateThreshold:  defaultCacheHitRateLow,
		QueueDeferThreshold:    defaultQueueDeferThreshold,
		DeferDuration:          "10ms",
		PrefixBaseWeight:       defaultPrefixBaseWeight,
		LoadBaseWeight:         defaultLoadBaseWeight,
		ActiveBaseWeight:       defaultActiveBaseWeight,
		EnableWorkloadProfiles: true,
		EnableOnlineAdaptation: true,
		OnlineDecayWindow:      defaultOnlineDecayWindow,
		TTFTThreshold:          defaultTTFTThreshold,
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

	kalman := &KalmanMetricTracker{
		QueueVariance: NewKalmanFilter1D(params.QueueProcessNoise, params.QueueMeasureNoise),
		KVUtilSpread:  NewKalmanFilter1D(params.KVProcessNoise, params.KVMeasureNoise),
		CacheHitRate:  NewKalmanFilter1D(params.CacheProcessNoise, params.CacheMeasureNoise),
	}

	return &AdaptiveScorer{
		typedName: plugin.TypedName{Type: AdaptiveScorerType, Name: name},
		params:    params,
		deferDur:  deferDur,
		state: &adaptiveState{
			kalman:        kalman,
			prefixWeight:  params.PrefixBaseWeight,
			loadWeight:    params.LoadBaseWeight,
			activeWeight:  params.ActiveBaseWeight,
			profileCounts: make(map[string]uint64),
			endpointPerfs: make(map[string]*endpointPerf),
			perfWindow:    params.OnlineDecayWindow,
		},
	}, nil
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

func (s *AdaptiveScorer) classifyWorkload(request *scheduling.LLMRequest) workloadProfile {
	if !s.params.EnableWorkloadProfiles || request == nil || request.Body == nil {
		return defaultAdaptiveProfile
	}
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
	if promptLen == 0 && request.RequestSizeBytes > 0 {
		promptLen = request.RequestSizeBytes
	}
	if promptLen > longContextPromptChars {
		return longContextProfile
	}
	if isChatCompletion {
		return conversationalProfile
	}
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

	profile := s.classifyWorkload(request)
	s.state.mu.Lock()
	s.state.profileCounts[profile.name]++
	s.state.mu.Unlock()

	// ── Observe ──
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
		kvUtils = append(kvUtils, float64(m.KVCacheUsagePercent)/100.0)
	}

	// ── Defer ──
	if profile.deferThreshold > 0 && minQueue > profile.deferThreshold {
		s.state.mu.Lock()
		s.state.deferCount++
		s.state.mu.Unlock()
		logger.Info("Adaptive: deferring", "profile", profile.name,
			"minQueue", minQueue, "threshold", profile.deferThreshold)
		time.Sleep(s.deferDur)
	}

	// ── Kalman update + weight adjustment ──
	s.state.mu.Lock()
	fQueueVar, fKVSpread, _ := s.state.kalman.Update(
		variance(queueDepths), spread(kvUtils), 0)
	s.state.totalRequests++

	prefixW := profile.prefixWeight
	loadW := profile.loadWeight
	activeW := profile.activeWeight

	if fQueueVar > s.params.QueueVarianceThreshold {
		boost := 1.0 + (fQueueVar-s.params.QueueVarianceThreshold)/s.params.QueueVarianceThreshold
		loadW *= boost
		activeW *= boost
		prefixW *= 0.7
	}
	if fKVSpread > s.params.KVUtilSpreadThreshold {
		loadW *= 1.3
	}
	if s.state.kalman.CacheHitRate.X < s.params.CacheHitRateThreshold && s.state.totalRequests > 100 {
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

	// ── Score ──
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
		if r := float64(m.RunningRequestsSize); r > 0 {
			activeScore = math.Max(0, 1.0-r/64.0)
		}
		composite := prefixW*prefixScore + loadW*loadScore + activeW*activeScore

		// ── Online adaptation: penalize endpoints with high observed TTFT ──
		if s.params.EnableOnlineAdaptation {
			epName := ep.GetMetadata().NamespacedName.String()
			s.state.mu.RLock()
			perf, exists := s.state.endpointPerfs[epName]
			s.state.mu.RUnlock()
			if exists && perf.count > 10 {
				// Sigmoid penalty: score *= 1/(1+exp(5*(meanTTFT - threshold)))
				// Endpoints with TTFT < threshold are unpenalized.
				// Endpoints with TTFT > threshold are progressively downscored.
				penalty := 1.0 / (1.0 + math.Exp(5*(perf.meanTTFT()-s.params.TTFTThreshold)))
				composite *= penalty
			}
		}
		scoredEndpoints[ep] = composite
	}

	if s.state.totalRequests%100 == 0 {
		s.state.mu.RLock()
		qGain := s.state.kalman.QueueVariance.Gain()
		kvGain := s.state.kalman.KVUtilSpread.Gain()
		logger.Info("Adaptive scorer",
			"profile", profile.name,
			"weights", fmt.Sprintf("p=%.3f l=%.3f a=%.3f", prefixW, loadW, activeW),
			"kalman", fmt.Sprintf("qVar=%.2f(K=%.2f) kvS=%.2f(K=%.2f)", fQueueVar, qGain, fKVSpread, kvGain),
			"defers", s.state.deferCount,
			"profiles", s.state.profileCounts)
		s.state.mu.RUnlock()
	}

	return scoredEndpoints
}

// RecordEndpointTTFT feeds TTFT observations for online adaptation.
// Should be called from a ResponseBody plugin after each response.
func (s *AdaptiveScorer) RecordEndpointTTFT(endpointName string, ttft float64) {
	if !s.params.EnableOnlineAdaptation {
		return
	}
	s.state.mu.Lock()
	defer s.state.mu.Unlock()
	perf, exists := s.state.endpointPerfs[endpointName]
	if !exists {
		perf = newEndpointPerf(s.state.perfWindow)
		s.state.endpointPerfs[endpointName] = perf
	}
	perf.record(ttft)
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

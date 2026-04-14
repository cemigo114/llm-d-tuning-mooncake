#!/usr/bin/env bash
set -euo pipefail

# Build the adaptive EPP scorer image in-cluster using a Go builder pod
# and push to ttl.sh (anonymous temporary registry, 24h expiry).
#
# Prerequisites:
#   - KUBECONFIG set
#   - llm-d-inference-scheduler source with adaptive.go at SCORER_SRC
#
# Output: prints IMAGE=<tag> on success

NAMESPACE="${NAMESPACE:-kpouget-dev}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCORER_SRC="${SCORER_SRC:-$SCRIPT_DIR/../adaptive-scorer}"
SCHEDULER_REPO="${SCHEDULER_REPO:-}"
IMAGE_TAG="${IMAGE_TAG:-ttl.sh/llm-d-epp-adaptive-$(date +%s):24h}"

echo "=== Building adaptive EPP scorer ==="
echo "  Namespace: $NAMESPACE"
echo "  Target image: $IMAGE_TAG"

# Step 1: Get the scheduler source
if [ -n "$SCHEDULER_REPO" ] && [ -d "$SCHEDULER_REPO" ]; then
  SRC_DIR="$SCHEDULER_REPO"
  echo "  Using existing repo: $SRC_DIR"
else
  SRC_DIR=$(mktemp -d)
  echo "  Cloning llm-d-inference-scheduler..."
  git clone --depth 1 https://github.com/llm-d/llm-d-inference-scheduler.git "$SRC_DIR"
fi

# Step 2: Copy adaptive scorer files into the scheduler source
echo "  Injecting adaptive scorer..."
cp "$SCORER_SRC/adaptive.go" "$SRC_DIR/pkg/plugins/scorer/adaptive.go"
cp "$SCORER_SRC/adaptive_test.go" "$SRC_DIR/pkg/plugins/scorer/adaptive_test.go"

# Register the plugin if not already registered
if ! grep -q "AdaptiveScorerType" "$SRC_DIR/pkg/plugins/register.go"; then
  sed -i.bak '/NoHitLRUFactory/a\
	plugin.Register(scorer.AdaptiveScorerType, scorer.AdaptiveScorerFactory)' \
    "$SRC_DIR/pkg/plugins/register.go"
  rm -f "$SRC_DIR/pkg/plugins/register.go.bak"
  echo "  Registered adaptive scorer in register.go"
fi

# Step 3: Create source tarball
TARBALL=$(mktemp /tmp/epp-src-XXXXXX.tar.gz)
tar czf "$TARBALL" -C "$SRC_DIR" --exclude='.git' .

# Step 4: Create builder pod
echo "  Creating builder pod..."
kubectl apply -n "$NAMESPACE" -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: epp-builder
  namespace: $NAMESPACE
spec:
  restartPolicy: Never
  containers:
  - name: builder
    image: quay.io/projectquay/golang:1.25
    command: ["sleep", "1800"]
    volumeMounts:
    - name: workspace
      mountPath: /workspace
  volumes:
  - name: workspace
    emptyDir:
      sizeLimit: 4Gi
EOF

kubectl wait --for=condition=Ready pod/epp-builder -n "$NAMESPACE" --timeout=120s

# Step 5: Copy source and build
echo "  Copying source..."
kubectl cp "$TARBALL" "$NAMESPACE/epp-builder:/workspace/src.tar.gz"

echo "  Building and pushing..."
kubectl exec epp-builder -n "$NAMESPACE" -- bash -c "
cd /workspace && tar xzf src.tar.gz 2>/dev/null

echo '--- go build ---'
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags='-s -w' -o bin/epp cmd/epp/main.go
if [ \$? -ne 0 ]; then echo 'BUILD FAILED'; exit 1; fi

echo '--- tests ---'
go test ./pkg/plugins/scorer/ -run 'TestAdaptive|TestVariance|TestSpread|TestClamp' -count=1
if [ \$? -ne 0 ]; then echo 'TESTS FAILED'; exit 1; fi

echo '--- push image ---'
curl -sL 'https://github.com/google/go-containerregistry/releases/download/v0.20.3/go-containerregistry_Linux_x86_64.tar.gz' | tar xz -C /usr/local/bin crane
mkdir -p /tmp/imgbuild && cp bin/epp /tmp/imgbuild/
crane append \
  --base gcr.io/distroless/static:nonroot \
  --new_layer <(cd /tmp/imgbuild && tar cf - --transform 's,^,app/,' epp) \
  --new_tag '$IMAGE_TAG'
echo 'IMAGE=$IMAGE_TAG'
"

# Step 6: Cleanup
echo "  Cleaning up builder pod..."
kubectl delete pod epp-builder -n "$NAMESPACE" --wait=false
rm -f "$TARBALL"
[ -z "$SCHEDULER_REPO" ] && rm -rf "$SRC_DIR"

echo ""
echo "=== Build complete ==="
echo "IMAGE=$IMAGE_TAG"
echo ""
echo "Deploy with:"
echo "  ./deploy-adaptive-epp.sh $IMAGE_TAG"

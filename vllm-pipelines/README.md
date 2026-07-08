# vLLM Pipelines

Build and deploy [fax4ever/vllm](https://github.com/fax4ever/vllm) on OpenShift, serving on the cluster's GPU.

## Next steps for you to run

### 1. Create the project

```bash
oc new-project vllm-pipelines
oc label namespace vllm-pipelines opendatahub.io/dashboard=true
```

### 2. Apply the build resources

```bash
oc apply -f openshift/image-build/imagestream.yaml
oc apply -f openshift/image-build/buildconfig.yaml
```

### 3. Build the serving image

Always built from git (committed + pushed to your fork's `main`) — the real, traceable serving artifact. For local edits, see step 5.

```bash
oc start-build vllm-fax4ever --follow
```

### 4. Serve the built image

Apply the manifests directly:

```bash
oc apply -f openshift/serving/servingruntime.yaml
oc apply -f openshift/serving/inferenceservice.yaml
```

Avoid the OpenShift AI console's deploy wizard for this: its "Connection" field for storage sets an explicit `imagePullSecrets` on the `InferenceService`, which blocks Kubernetes from auto-merging the `default` service account's own internal-registry pull secret and breaks the `vllm-fax4ever:latest` image pull with an "authentication required" error — even though the modelcar image is public and never needed a connection to begin with.

### 5. Test local Python changes on the GPU

Only for pure-Python changes — C++/CUDA (`.cu`/`csrc`) changes are out of scope.

One-time setup — build the debug image (editable install of your checkout + pytest) and start the debug pod:

```bash
oc apply -f openshift/debug/imagestream.yaml
oc apply -f openshift/debug/buildconfig.yaml
oc start-build vllm-fax4ever-patched --from-dir=/home/fax/code/vllm --follow \
  --exclude='(^|/)(\.git|\.venv|dist|build|__pycache__|\.mypy_cache)(/|$)'

oc apply -f openshift/debug/deployment.yaml
oc rollout status deployment/vllm-fax4ever-debug -n vllm-pipelines
```

Fast loop — repeat for every edit, no rebuild needed:

```bash
POD=$(oc get pod -n vllm-pipelines -l app=vllm-fax4ever-debug -o jsonpath='{.items[0].metadata.name}')

oc rsync --no-perms /home/fax/code/vllm/vllm/ vllm-pipelines/$POD:/workspace/vllm-src/vllm/
oc exec -n vllm-pipelines $POD -- python3 -m pytest /workspace/vllm-src/tests/kernels/ -v
```

```bash
# free the GPU when you're done for now (without deleting anything)
oc scale deployment/vllm-fax4ever-debug -n vllm-pipelines --replicas=0
```

## Alternative: vanilla Kubernetes (no OpenShift AI/KServe)

`k8s/deployment.yaml` is a plain `Deployment`/`Service`, no RHOAI/KServe dependency — works on any Kubernetes cluster with a GPU device plugin exposing `nvidia.com/gpu`. Requires the image to be pushed to a registry reachable from that cluster (edit the `image:` field) since the OpenShift internal registry route won't be reachable elsewhere.

```bash
kubectl apply -f k8s/deployment.yaml
```

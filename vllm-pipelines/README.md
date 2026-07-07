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

### 3. Trigger a build

Option A: build from git main

```bash
oc start-build vllm-fax4ever --follow
```

Option B: build from local uncommitted changes

```bash
oc start-build vllm-fax4ever --from-dir=/home/fax/code/vllm --follow
```

### 4. Serve the built image

Option A: apply the manifests

```bash
oc apply -f openshift/serving/servingruntime.yaml
oc apply -f openshift/serving/inferenceservice.yaml
```

Option B: use the OpenShift AI console

1. **Settings → Serving runtimes → Add serving runtime**, paste the contents of `openshift/serving/servingruntime.yaml`.
2. In the `vllm-pipelines` project, **Models → Deploy model**, select the `vllm-fax4ever-runtime` runtime just added.
3. Set the connection/storage location to `oci://quay.io/redhat-ai-services/modelcar-catalog:llama-3.2-3b-instruct`, then deploy.

### 5. Test local Python changes on the GPU

Only for pure-Python changes — C++/CUDA (`.cu`/`csrc`) changes are out of scope.

`VLLM_USE_PRECOMPILED=1` (step 3) means rebuilding `vllm-fax4ever` from your local checkout is now fast and already picks up your latest Python edits. So: rebuild that first, then layer on `pytest` + the `tests/` directory (not shipped in the runtime image) and run the test as a one-shot `Job`:

```bash
# 1. rebuild the base image with your latest local changes
oc start-build vllm-fax4ever --from-dir=/home/fax/code/vllm --follow

# 2. add pytest + tests/, then run the test
oc apply -f openshift/debug/imagestream.yaml
oc apply -f openshift/debug/buildconfig.yaml
oc start-build vllm-fax4ever-patched --from-dir=/home/fax/code/vllm --follow

oc apply -f openshift/debug/deployment.yaml
oc rollout status deployment/vllm-fax4ever-debug -n vllm-pipelines

# run all kernel tests
oc exec -n vllm-pipelines deployment/vllm-fax4ever-debug -- python3 -m pytest /tmp/tests/kernels/ -v

# or target a specific test/pattern
oc exec -n vllm-pipelines deployment/vllm-fax4ever-debug -- python3 -m pytest /tmp/tests/kernels/test_compressor_kv_cache.py -v

# free the GPU when you're done for now (without deleting anything)
oc scale deployment/vllm-fax4ever-debug -n vllm-pipelines --replicas=0
```

## Alternative: vanilla Kubernetes (no OpenShift AI/KServe)

`k8s/deployment.yaml` is a plain `Deployment`/`Service`, no RHOAI/KServe dependency — works on any Kubernetes cluster with a GPU device plugin exposing `nvidia.com/gpu`. Requires the image to be pushed to a registry reachable from that cluster (edit the `image:` field) since the OpenShift internal registry route won't be reachable elsewhere.

```bash
kubectl apply -f k8s/deployment.yaml
```

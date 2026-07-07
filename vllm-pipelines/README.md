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
oc apply -f openshift/build/imagestream.yaml
oc apply -f openshift/build/buildconfig.yaml
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

## Alternative: vanilla Kubernetes (no OpenShift AI/KServe)

`k8s/deployment.yaml` is a plain `Deployment`/`Service`, no RHOAI/KServe dependency — works on any Kubernetes cluster with a GPU device plugin exposing `nvidia.com/gpu`. Requires the image to be pushed to a registry reachable from that cluster (edit the `image:` field) since the OpenShift internal registry route won't be reachable elsewhere.

```bash
kubectl apply -f k8s/deployment.yaml
```

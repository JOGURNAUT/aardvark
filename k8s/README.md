# Running Aardvark on Kubernetes

The deployed copy runs on Azure Container Apps, which is a managed layer over
the same primitives. These manifests are the primitives written out, which is
useful for one reason: the managed layer hides the two design decisions in this
application that do not survive being scaled, and Kubernetes makes you write
them down.

## Run it locally

```bash
kind create cluster --config k8s/kind-cluster.yaml

docker build -t aardvark:local .
kind load docker-image aardvark:local          # kind has its own image store

kubectl create secret generic aardvark-secrets --from-env-file=.env
kubectl apply -f k8s/configmap.yaml -f k8s/pvc.yaml \
               -f k8s/deployment.yaml -f k8s/service.yaml \
               -f k8s/service-nodeport.yaml

kubectl rollout status deploy/aardvark
```

Then <http://localhost:8080>.

`hpa.yaml` is deliberately not in that list. See **What blocks scaling** below.

## What changes, against Azure today

| | Azure Container Apps now | These manifests |
|---|---|---|
| First request after a cold start | times out, second one works | held at the Service until `/api/ready` is green |
| Process hangs | noticed on the next request | `livenessProbe` restarts it within ~60s |
| Chat history | on the container filesystem, gone on restart | on a PersistentVolumeClaim, survives |
| Memory ceiling | platform default | 3Gi limit, declared |
| Deploy | new revision, platform decides the swap | `Recreate`, and the reason is in the file |
| Scaling | scale-to-zero, one replica | still one replica, and now the reason is written down |

The first row is the one worth having. It is visible in the deployment we
already run: a request to the live URL after it has scaled to zero times out,
and the retry succeeds. That is not a network fault, it is the embedding model
being a lazy global in `agent/selector.py`, loaded on the first query rather
than at startup.

`api/server.py` now warms the model on a background thread and reports it on
`/api/ready`, which stays 503 until it finishes. `/api/health` answers as soon
as HTTP is up and is used for liveness only. Gating liveness on the model would
restart a pod that is merely still warming, and turn a slow start into a crash
loop.

## What blocks scaling

One thing, and it is not the one this file first claimed.

**It is not session state.** The FastAPI path keeps nothing per-session in
memory: the client holds the session id and every turn is read back from
SQLite. Two pods behind the Service, 40 sessions created through it, both pods
then read all 40. The Streamlit UI in `ui/app.py` *does* keep state in the
process, which is why it was always run at one replica, but it is no longer
what the image starts. `service.yaml` had `sessionAffinity: ClientIP` on that
mistaken basis and no longer does.

**It is the ReadWriteOnce volume.** Two pods on the same node share it without
complaint, because POSIX locks work correctly on a local filesystem: the same
40-write test produced no lock errors. Two pods on different nodes cannot. A
ReadWriteOnce volume is mountable by one node at a time, so the second node's
pod never schedules, and giving it a ReadWriteMany volume instead puts SQLite
on a network filesystem, where its locking is advisory and two writers corrupt
rather than contend.

That is why `replicas: 1` and `strategy: Recreate` are here even though both
are unnecessary on a single node. They are the behaviour that is correct on
either topology, and a Deployment should not be one added node away from
breaking.

**One change removes all of it:** move the chat history from SQLite to
Postgres. Then the volume goes, `replicas > 1` is safe anywhere,
`RollingUpdate` replaces `Recreate`, and `hpa.yaml` applies unchanged.

## What is deliberately not here

- **No `securityContext`.** `runAsNonRoot` is correct and it breaks this image:
  the embedding model is baked in at build time into root's Hugging Face cache,
  so a non-root container cannot find it and re-downloads 80 MB on every cold
  start. The fix is `HF_HOME` at build time, then the securityContext. Adding
  the securityContext first would move the failure rather than remove it.
- **No Ingress or TLS.** NodePort is for the local cluster. A real cluster
  terminates TLS at an ingress controller.
- **No ServiceMonitor.** Per-stage latency already goes to SQLite; exporting it
  as Prometheus metrics is the obvious next step and is not done.
- **No resource limit on CPU.** A CPU limit throttles rather than kills, and
  throttling the embedding pass makes every answer slower for no benefit.

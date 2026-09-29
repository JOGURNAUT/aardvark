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

Two things, both in the application rather than the manifests.

**Session state lives in the server process.** A second replica answers a
follow-up question with no memory of the conversation it is following up on.
`service.yaml` sets `sessionAffinity: ClientIP`, which is a mitigation and not
a fix: it breaks behind a NAT or any proxy that rewrites the source address,
and it pins a user to a pod that may be rescheduled underneath them.

**Chat history is SQLite on a ReadWriteOnce volume.** One node can mount that
for writing at a time, which is what SQLite can survive: its locking is built
on POSIX file locks, which are advisory and unreliable over shared filesystems,
so two writers corrupt the database rather than contend for it.

Those two facts produce every other constraint here. `replicas: 1`.
`strategy: Recreate`, because a rolling update would deadlock waiting for a
volume the outgoing pod still holds. `hpa.yaml` sitting unapplied.

The order to fix them in:

1. **Session state into Redis**, or a signed cookie. Unblocks `replicas > 1`.
2. **Chat history into Postgres.** Unblocks `RollingUpdate` and the HPA.

Neither is large. What is worth noticing is that both were reasonable
single-replica decisions that only became visible as constraints when
something asked for a second replica.

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

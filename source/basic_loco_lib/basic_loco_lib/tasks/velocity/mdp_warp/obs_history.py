"""Per-term observation history for the Warp observation manager.

``make_history_obs`` wraps a source term with a device-side ring buffer and
emits its frames from oldest to newest. Concatenating wrapped terms preserves
the per-term history layout used by the Torch observation manager.

Uniform noise is applied once to each incoming frame. Terminated environments
have their history filled with the fresh frame, matching CircularBuffer. If the
termination mask is unavailable, a warning is issued and clearing is disabled.

Buffers must be initialized before CUDA graph capture; counters and noise RNG
state then advance on the device. Frame dimensions are inferred from source
metadata or resolved against the live environment during manager setup.
"""

from __future__ import annotations

import copy
import inspect
import weakref

import torch
import warp as wp

from isaaclab_experimental.envs.utils.io_descriptors import generic_io_descriptor_warp

__all__ = ["make_history_obs", "register_env_hint", "resolve_dones"]

# Weak reference to the env most recently under construction, registered by
# MotionCommandWarp.__init__.  ManagerBasedRLEnvWarp builds the command
# manager BEFORE the observation manager (with the sim already playing), so
# when a deferred descriptor resolves the wrapper's out_dim during
# ObservationManager init, the live env is reachable from here.
_env_hint: weakref.ReferenceType | None = None


def register_env_hint(env) -> None:
    """Record the env being constructed for deferred obs-dim resolution.

    Weakref on purpose: the hint must not keep a destroyed env alive, and
    each new env construction (train / play / parity) simply refreshes it.
    """
    global _env_hint
    _env_hint = weakref.ref(env)


@wp.kernel
def _tick_kernel(counter: wp.array(dtype=wp.int32)):
    counter[0] = counter[0] + 1


@wp.kernel
def _history_push_flatten_kernel(
    scratch: wp.array(dtype=wp.float32, ndim=2),  # (N, D) clean current frame
    ring: wp.array(dtype=wp.float32, ndim=3),  # (N, H, D)
    counter: wp.array(dtype=wp.int32),  # (1,) frames pushed so far = index of current frame
    rng_state: wp.array(dtype=wp.int32, ndim=2),  # (N, D) per-element RNG streams
    dones: wp.array(dtype=wp.bool),  # (N,) env terminated THIS step (reset already applied)
    noise_lo: float,
    noise_hi: float,
    history_length: int,
    frame_dim: int,
    out: wp.array(dtype=wp.float32, ndim=2),  # (N, H*D) chronological old -> new
):
    i, j = wp.tid()
    t = counter[0]
    state = wp.uint32(rng_state[i, j])
    value = scratch[i, j] + wp.randf(state, noise_lo, noise_hi)
    rng_state[i, j] = wp.int32(state + wp.uint32(1))
    # One thread owns all history slots for an env/feature, so backfill and
    # normal pushes cannot race. Noise is sampled once, including on reset.
    if t == 0 or dones[i]:
        for slot in range(history_length):
            ring[i, slot, j] = value
    else:
        ring[i, t % history_length, j] = value
    for k in range(history_length):
        slot = (t - (history_length - 1 - k)) % history_length
        if slot < 0:
            slot += history_length
        out[i, k * frame_dim + j] = ring[i, slot, j]


def _as_wp_bool(view, num_envs: int):
    """Coerce a dones-like object to a contiguous (num_envs,) wp.bool array."""
    try:
        if isinstance(view, wp.array):
            arr = view
        elif torch.is_tensor(view):
            if view.dtype != torch.bool or not view.is_contiguous():
                return None
            arr = wp.from_torch(view)
        else:
            # lazy warp-backed buffer (Newton): round-trip like MotionCommandWarp._live_wp
            t = wp.to_torch(view)
            if t.dtype != torch.bool or not t.is_contiguous():
                return None
            arr = wp.from_torch(t)
        if arr.dtype != wp.bool or tuple(arr.shape) != (num_envs,):
            return None
        return arr
    except Exception:
        return None


def resolve_dones(env):
    """Device-readable per-env "terminated this step" mask, or ``None``.

    Resolved per call (not cached): in eager mode the manager may rebind the
    buffer, and under graph capture the Python body only runs at capture time
    anyway, so the pointer seen there is what gets baked in.  The
    experimental termination terms write ``wp.bool`` outs (see
    ``mdp_warp/terminations.py``), so the aggregated dones is expected to be
    a stable ``(num_envs,)`` wp.bool buffer written earlier in the same
    captured step.
    """
    term_manager = getattr(env, "termination_manager", None)
    return _as_wp_bool(getattr(term_manager, "dones", None), env.num_envs)


def _iter_obs_terms(env):
    """Yield ``(group_cfg, term_cfg)`` for every obs term of ``env.cfg``.

    These are the SAME objects the managers hold and mutate
    (``ManagerBasedEnvWarp`` stores ``self.cfg = cfg`` without copying), so
    by dim-inference time a term's materialized signature defaults and
    resolved SceneEntityCfgs are visible here.  Defensive: any
    introspection failure yields nothing.
    """
    try:
        for group in vars(env.cfg.observations).values():
            terms = None
            get_terms = getattr(group, "get_terms", None)
            if callable(get_terms):
                try:
                    terms = dict(get_terms())
                except Exception:
                    terms = None
            if terms is None:
                terms = {k: v for k, v in vars(group).items() if hasattr(v, "func")}
            for term in terms.values():
                yield group, term
    except Exception:
        pass


def _detect_group_corruption(env, term_func, default: bool = True) -> bool:
    """Find the obs group owning ``term_func`` and return its enable_corruption.

    Defensive: any introspection failure falls back to ``default``.  Runs
    once at init (host-side) and the answer is baked into the captured
    kernels -- corruption state does not change within an env instance.
    """
    for group, term in _iter_obs_terms(env):
        if getattr(term, "func", None) is term_func:
            return bool(getattr(group, "enable_corruption", default))
    return default


def _find_term_params(env, term_func) -> dict:
    """The ``params`` of the ObsTerm owning ``term_func`` ({} if not found)."""
    for _, term in _iter_obs_terms(env):
        if getattr(term, "func", None) is term_func:
            return dict(getattr(term, "params", None) or {})
    return {}


class _HistoryState:
    """Per-env persistent buffers of one wrapped term (allocated once, eager)."""

    __slots__ = (
        "scratch", "ring", "counter", "rng_state", "dones_fallback",
        "num_envs", "frame_dim", "warned", "lo", "hi", "params",
    )


def _init_state(env, frame_dim: int, history_length: int) -> _HistoryState:
    if wp.get_device(env.device).is_capturing:
        raise RuntimeError(
            "make_history_obs: first invocation happened during CUDA-graph capture. "
            "The persistent history buffers must be allocated on an eager (warmup) step; "
            "a one-time init recorded into the graph would replay every step."
        )
    n, d, h = env.num_envs, frame_dim, history_length
    dev = env.device
    st = _HistoryState()
    st.num_envs = n
    st.frame_dim = d
    st.warned = False
    # persistent torch storage + one-time zero-copy warp views (stable pointers)
    st.scratch = wp.from_torch(torch.zeros(n, d, dtype=torch.float32, device=dev))
    st.ring = wp.from_torch(torch.zeros(n, h, d, dtype=torch.float32, device=dev))
    st.counter = wp.from_torch(torch.zeros(1, dtype=torch.int32, device=dev))
    # per-element RNG streams: every wrapped term draws its own at init, so
    # streams are decorrelated across terms (and reproducible under a seeded
    # run -- the same global-torch-RNG dependency Unoise sampling has)
    rng = torch.randint(0, 2**31 - 1, (n, d), dtype=torch.int64, device=dev).to(torch.int32)
    st.rng_state = wp.from_torch(rng)
    # used when the termination manager exposes no readable dones buffer
    st.dones_fallback = wp.from_torch(torch.zeros(n, dtype=torch.bool, device=dev))
    return st


def _resolve_source_params(env, source_func, source_params: dict) -> dict:
    """Materialize + resolve the source term's signature defaults.

    Mirrors what the manager does for the terms it sees:
    ``ManagerBase._resolve_common_term_cfg`` copies dataclass defaults from
    the signature into params, and ``_process_term_cfg_at_play`` calls
    ``SceneEntityCfg.resolve(scene)`` on them (filling ``joint_ids`` /
    ``joint_ids_wp``).  The manager only inspects the wrapper -- whose
    signature has no such defaults -- so the wrapped source term (e.g. the
    official ``joint_pos_rel``, which raises without a resolved
    ``asset_cfg.joint_ids_wp``) must be resolved here.  Runs once on the
    first eager call, when the simulation is playing and scene entities are
    resolvable; the resolved id buffers are pointer-stable for graph replay.
    """
    resolved = dict(source_params)
    try:
        signature = inspect.signature(source_func)
    except (TypeError, ValueError):
        return resolved
    for param in list(signature.parameters.values())[2:]:  # skip (env, out)
        if param.default is inspect.Parameter.empty:
            continue
        if param.name in resolved or not hasattr(param.default, "__dataclass_fields__"):
            continue
        default = param.default
        resolved[param.name] = default.copy() if hasattr(default, "copy") else copy.deepcopy(default)
    for value in resolved.values():
        # SceneEntityCfg-style: resolve against the live scene (duck-typed to
        # stay import-agnostic; same shape the manager isinstance-checks)
        if hasattr(value, "resolve") and hasattr(value, "joint_names") and hasattr(value, "name"):
            value.resolve(env.scene)
    return resolved


def _get_descriptor(source_func):
    """The source term's io-descriptor (through functools.wraps chains), or None."""
    descriptor = getattr(source_func, "_descriptor", None)
    if descriptor is None:
        descriptor = getattr(getattr(source_func, "__wrapped__", None), "_descriptor", None)
    return descriptor


def _infer_frame_dim(source_func) -> int | None:
    """Cfg-time frame dim from the source term's io-descriptor, or ``None``.

    Mirrors the first two steps of the manager's ``_infer_term_dim_scalar``:
    a concrete ``out_dim`` int, else ``len(axes)``.  ``None`` means the dim
    only exists at runtime (sentinels ``"command"`` / ``"joint"`` /
    ``"action"`` / ``"body:N"``, or no descriptor metadata at all like the
    joint-state terms) -- ``make_history_obs`` then DEFERS the resolution to
    ``ObservationManager`` init via :class:`_DeferredDim`.
    """
    descriptor = _get_descriptor(source_func)
    if descriptor is None:
        return None
    out_dim = getattr(descriptor, "out_dim", None)
    if isinstance(out_dim, int):
        return int(out_dim)
    extras = getattr(descriptor, "extras", None)
    axes = extras.get("axes") if extras else None
    if axes is not None:
        return len(axes)
    return None


def _resolve_frame_dim_live(env, source_func, params: dict) -> int:
    """Per-frame dim of ``source_func`` resolved against a LIVE env.

    Task-side mirror of the manager's ``_infer_term_dim_scalar`` /
    ``_resolve_out_dim`` for the sources we wrap: descriptor int, then the
    sentinels (``"command"`` / ``"action"`` / ``"body:N"`` / ``"joint"``),
    then ``axes``, then the ``asset_cfg`` joint-count fallback.  ``params``
    must be the term's own params (materialized signature defaults +
    resolved SceneEntityCfgs) -- exactly what the manager has when it
    infers dims (sim is playing at ObservationManager init and
    ``_resolve_common_term_cfg`` ran for this term), and what
    ``_resolve_source_params`` produces for direct calls.
    """
    descriptor = _get_descriptor(source_func)
    out_dim = getattr(descriptor, "out_dim", None) if descriptor is not None else None
    if isinstance(out_dim, int):
        return int(out_dim)
    if out_dim == "command":
        cmd = env.command_manager.get_command(params.get("command_name"))
        return int(cmd.shape[-1])
    if out_dim == "action":
        return int(env.action_manager.action.shape[-1])
    if isinstance(out_dim, str) and out_dim.startswith("body:"):
        per_body = int(out_dim.split(":")[1])
        asset_cfg = params.get("asset_cfg")
        asset = env.scene[asset_cfg.name]
        body_ids = getattr(asset_cfg, "body_ids", None)
        if body_ids is None or isinstance(body_ids, slice):
            return per_body * len(asset.body_names)
        return per_body * len(body_ids)
    if out_dim == "joint" or out_dim is None:
        asset_cfg = params.get("asset_cfg")
        if asset_cfg is not None:
            joint_ids_wp = getattr(asset_cfg, "joint_ids_wp", None)
            if joint_ids_wp is not None:
                return int(joint_ids_wp.shape[0])
            joint_ids = getattr(asset_cfg, "joint_ids", slice(None))
            if isinstance(joint_ids, slice):
                return int(len(env.scene[asset_cfg.name].joint_names))
            return int(len(joint_ids))
    extras = getattr(descriptor, "extras", None) if descriptor is not None else None
    axes = extras.get("axes") if extras else None
    if axes is not None:
        return len(axes)
    raise ValueError(
        f"make_history_obs: cannot resolve the frame dim of "
        f"{getattr(source_func, '__name__', source_func)!r} against the live env "
        f"(descriptor out_dim={out_dim!r}, no axes, no asset_cfg in params) -- "
        "pass frame_dim= explicitly."
    )


class _DeferredDim:
    """Descriptor proxy whose ``out_dim`` resolves to ``frame_dim * H`` lazily.

    Installed as ``wrapper._descriptor`` when the frame dim is not knowable
    at cfg time.  The manager reads ``descriptor.out_dim`` during
    ``ObservationManager`` init (``_infer_term_dim_scalar`` step 1, a plain
    ``getattr`` -- no isinstance check on the descriptor), at which point
    the sim is playing and the command/action managers exist, so the
    source's sentinel dim resolves exactly like the framework resolves it
    for unwrapped terms -- then gets multiplied by the history length,
    which the framework resolvers cannot express.  All other descriptor
    attributes pass through to the real descriptor.
    """

    __slots__ = ("_inner", "_wrapper", "_source_func", "_history_length", "_cached")

    def __init__(self, inner, wrapper, source_func, history_length: int):
        self._inner = inner
        self._wrapper = wrapper
        self._source_func = source_func
        self._history_length = history_length
        self._cached = None

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def out_dim(self):
        if self._cached is None:
            env = _env_hint() if _env_hint is not None else None
            if env is None:
                raise ValueError(
                    f"make_history_obs: no live env registered to resolve the deferred frame dim of "
                    f"{getattr(self._source_func, '__name__', self._source_func)!r} (the hint is set by "
                    "MotionCommandWarp.__init__, which runs before the ObservationManager inits). "
                    "Pass frame_dim= explicitly for tasks without a MotionCommandWarp command term."
                )
            # cfg-side params are the RAW ones: the manager materializes
            # signature defaults (asset_cfg) into its own term-cfg copies,
            # so run the same materialize+resolve mirror the wrapper body
            # uses for direct calls (deepcopy: resolve() mutates in place
            # and must not touch cfg-side objects)
            params = _resolve_source_params(
                env, self._source_func, copy.deepcopy(_find_term_params(env, self._wrapper))
            )
            frame_dim = _resolve_frame_dim_live(env, self._source_func, params)
            # publish the resolved value on the wrapper (tooling/debug aid)
            self._wrapper.frame_dim = frame_dim
            self._cached = frame_dim * self._history_length
        return self._cached


def _parse_noise(noise) -> tuple[float, float]:
    """``(lo, hi)`` of the uniform noise from a UniformNoiseCfg or a 2-tuple."""
    if noise is None:
        return 0.0, 0.0
    n_min, n_max = getattr(noise, "n_min", None), getattr(noise, "n_max", None)
    try:
        if n_min is not None and n_max is not None:
            # classic or experimental UniformNoiseCfg -- the same object the
            # non-history cfgs put on the ObsTerm's noise=
            return float(n_min), float(n_max)
        if isinstance(noise, (tuple, list)) and len(noise) == 2:
            return float(noise[0]), float(noise[1])
    except TypeError:
        pass
    raise ValueError(
        f"make_history_obs: unsupported noise cfg {noise!r} -- only uniform noise is "
        "supported (a UniformNoiseCfg with scalar n_min/n_max -- the same object the "
        "ObsTerm used to carry -- or an (n_min, n_max) tuple); the wrapper samples "
        "it in-kernel, once per frame, before the value enters the history ring."
    )


def make_history_obs(source_func, history_length: int = 4, frame_dim: int | None = None, noise=None):
    """Wrap a Warp obs term into a chronological ``history_length``-frame term.

    Args:
        source_func: the underlying per-step Warp term (e.g. any
            ``isaaclab_experimental.envs.mdp`` term or a ``mdp_warp``
            observation).  It must write the clean current frame into the
            provided ``(num_envs, frame_dim)`` buffer.  The ObsTerm's own
            ``params=`` reach it unchanged: the manager calls the wrapper
            with them as kwargs and the wrapper forwards them (merged over
            the source's signature defaults, which are materialized and
            resolved against the live scene on first call -- exactly what
            the manager does for the terms it sees directly).
        history_length: number of frames H kept and emitted (old -> new).
            Exposed as an attribute on the returned function -- ``play.py``
            reads it to build the deploy-side ONNX Gather permutation when
            the group-level ``history_length`` cfg knob is unavailable
            (Warp tasks).
        frame_dim: per-frame dimension of the source term.  Optional and
            normally omitted: concrete ``out_dim`` ints / ``axes`` infer at
            cfg time, and runtime sentinels (``"command"`` / ``"joint"`` /
            ``"action"``) or no-metadata joint terms resolve against the
            live env during ``ObservationManager`` init (deferred
            descriptor, see :class:`_DeferredDim`).  Pass explicitly only
            to bypass the deferred path (tasks with no ``MotionCommandWarp``
            term to register the env hint).
        noise: uniform noise applied ONCE per frame before it enters the
            ring (mirrors the torch manager's noise->buffer ordering): the
            same ``UniformNoiseCfg`` object the non-history cfgs put on the
            ObsTerm (``Unoise(n_min=..., n_max=...)``), or an
            ``(n_min, n_max)`` tuple.  ``None`` = corruption-free term.
            Keep ``noise=None`` on the ObsTerm itself: manager-level noise
            would hit the whole H x d output every step (re-noising stored
            frames), and the classic Unoise is silently skipped by the
            experimental manager anyway (isinstance mismatch).

    Returns:
        A ``generic_io_descriptor_warp``-decorated term function with
        ``out_dim = frame_dim * history_length``.
    """
    if frame_dim is None:
        frame_dim = _infer_frame_dim(source_func)  # None -> deferred to env init
    noise_lo, noise_hi = _parse_noise(noise)

    @generic_io_descriptor_warp(
        out_dim=None if frame_dim is None else frame_dim * history_length, dtype=torch.float32
    )
    def history_obs(env, out, **kwargs):
        states = history_obs._states
        state = states.get(id(env))
        if state is None or state.num_envs != env.num_envs:
            # resolve the owning group's corruption flag once (baked into the
            # captured kernels); noise=None terms stay clean regardless
            lo, hi = noise_lo, noise_hi
            if (lo != 0.0 or hi != 0.0) and not _detect_group_corruption(env, history_obs):
                lo, hi = 0.0, 0.0
            state_params = _resolve_source_params(env, source_func, {})
            d = frame_dim
            if d is None:
                # deferred dim, direct-call path (under the manager the
                # descriptor property already resolved it at init)
                d = _resolve_frame_dim_live(env, source_func, {**state_params, **kwargs})
                if history_obs.frame_dim is None:
                    history_obs.frame_dim = d
            state = _init_state(env, d, history_length)
            state.lo, state.hi = lo, hi
            state.params = state_params
            states[id(env)] = state
        # 1) clean current frame from the source term (its own kernel(s)):
        #    signature-default SceneEntityCfgs resolved once at init, plus
        #    the ObsTerm params= the manager forwards as kwargs every call
        source_func(env, state.scratch, **{**state.params, **kwargs})
        # 2) push + chronological flatten (device-side ring, race-free slots)
        dones = resolve_dones(env)
        if dones is None:
            dones = state.dones_fallback
            if not state.warned:
                print(
                    f"[WARN] obs_history/{history_obs.__name__}: termination_manager.dones not "
                    "readable as a (num_envs,) wp.bool buffer -- history reset-clearing disabled "
                    "(stale frames roll out within H-1 steps after a reset)."
                )
                state.warned = True
        wp.launch(
            kernel=_history_push_flatten_kernel,
            dim=(env.num_envs, state.frame_dim),
            inputs=[
                state.scratch,
                state.ring,
                state.counter,
                state.rng_state,
                dones,
                state.lo,
                state.hi,
                history_length,
                state.frame_dim,
                out,
            ],
            device=env.device,
        )
        # 3) advance the device frame counter (after all reads of it this step)
        wp.launch(kernel=_tick_kernel, dim=1, inputs=[state.counter], device=env.device)

    if frame_dim is None:
        # sentinel/no-metadata source: hand the manager a descriptor whose
        # out_dim resolves (frame_dim * H) against the live env at
        # ObservationManager init -- see _DeferredDim
        history_obs._descriptor = _DeferredDim(
            history_obs._descriptor, history_obs, source_func, history_length
        )
    history_obs._states = {}
    history_obs.frame_dim = frame_dim
    history_obs.history_length = history_length
    history_obs.source_func = source_func
    history_obs.__name__ = f"{getattr(source_func, '__name__', 'obs')}_h{history_length}"
    # Closures need a real module binding for config serialization. Keep every
    # wrapper distinct (policy/critic may wrap the same source with other noise).
    name = history_obs.__name__
    suffix = 1
    while name in globals():
        name = f"{history_obs.__name__}_{suffix}"
        suffix += 1
    history_obs.__name__ = name
    history_obs.__qualname__ = name
    history_obs.__module__ = __name__
    globals()[name] = history_obs
    # Present the SOURCE term's signature to the manager.  Its static check
    # (manager_base._resolve_common_term_cfg) rejects **kwargs passthrough --
    # VAR_KEYWORD has no default, so 'kwargs' lands in the mandatory list
    # ("expects mandatory parameters: ['kwargs']"; the framework flags this
    # case as unhandled right there).  With the source's signature exposed
    # via __signature__, the wrapper validates exactly like the unwrapped
    # term: the ObsTerm's params= pass the set check, dataclass defaults
    # (asset_cfg) are materialized into params and resolved at play time by
    # the manager itself, and the runtime call func(env, out_wp, **params)
    # is what the **kwargs body accepts and forwards.
    try:
        history_obs.__signature__ = inspect.signature(source_func)
    except (TypeError, ValueError):
        pass  # uninspectable source: keep the real (env, out, **kwargs)
    return history_obs
